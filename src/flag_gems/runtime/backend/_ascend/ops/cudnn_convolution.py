# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cudnn_convolution import _output_size, _to_list
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ascend direct path.
#
# Each program owns whole output rows rather than a slice of the flattened
# (n, oh, ow) space, so the inner tl.arange is the output column index and the
# input address ``iw = ow * SW + kw * DW`` is affine in it.  With unit stride
# that is a contiguous load.  Tiling the flattened space -- what the shared
# direct kernels do -- leaves the stride-1 index discontinuous at every row
# boundary, which turns each tap into a vector-indexed gather; the Ascend
# backend charges roughly four orders of magnitude for those, and it showed: the
# same shapes that run in 0.4 ms native took 388 ms.
#
# The address is affine *unpadded* because the padding is materialised instead:
# see _pad_input.  Anything that makes the address non-affine again -- a clamp
# to keep a masked lane in range, say -- costs more than the whole rest of the
# rewrite, which is why the halo is built rather than tested for.
#
# Within that tiling there are two kernels -- one per spatial rank -- and each
# carries both forms: a tl.dot branch and a plain FMA one, selected per call by
# _can_use_dot.
#
# * The tl.dot branch reduces a (BLOCK_C, BLOCK_W) input tile against a
#   (BLOCK_OC, BLOCK_C) weight tile, one dot per tap and channel block.  This is
#   what puts the operator on the cube, and msprof confirms it: the dot kernel
#   reports Task Type=MIX_AIC, while the FMA kernel below reports
#   AI_VECTOR_CORE.  On the 2D core case that is 3.29 ms against 28.0 ms.
#
#   The FMA form is not the more accurate one, which is worth stating plainly
#   because this file previously assumed the opposite.  Against an fp64 CPU
#   reference on (4,64,64,64)/3x3, the dot kernel's error is 1.068e-04 absolute
#   / 9.16e-07 relative -- numerically identical to the vendor conv run with
#   HF32 off, so it is the same exact-fp32 arithmetic, not the cube's
#   reduced-precision mode.  The FMA kernel is 3.052e-05 / 2.62e-07, 3.5x
#   tighter, but that is precision no fp32 tolerance asks for and it costs 3.9x
#   in time.  For scale, the vendor conv with HF32 *on* is 1.694e-02 / 1.45e-04,
#   which is where the ~11-bit-mantissa figure this file used to cite comes
#   from; tl.dot here does not do that.
#
# * The FMA form keeps one tap and one channel per iteration, for the shapes
#   tl.dot cannot express: any tile dimension below 16, i.e. every shape with
#   fewer than 16 output channels per group or fewer than 16 input channels.
#   Those are the small launch-bound cases, where padding up to 16 would
#   multiply a cheap kernel by 16/3.  It is what the two 3D shapes that miss the
#   dot minimum keep: stride 2 leaves an 8-wide output row, and C_in 4 is a
#   4-deep contraction, and both run it as fast as before the 3D arm was fixed.
#
# The dot path also has to fit the unified buffer, which is a correctness limit
# rather than a speed consideration: past it BiShengHIR either aborts the build
# with "ub overflow" or, for a tile that only just overruns, emits a kernel that
# faults the device.  _pick_block_c shrinks the channel tile to fit; the budget
# and the measurement behind it are at _UB_TILE_MAX.
#
# Both read the weight from the transposed (KH, KW, C, OC) layout built by
# _prep_weight.  That is not incidental: the output-channel axis is what the
# inner loop indexes, and it is a gather in the native layout and unit-stride in
# this one -- 140.8 ms against 36.1 ms with the kernel otherwise unchanged.
#
# Turning that tile a quarter turn -- (KH, KW, OC, C), so that the tile's *last*
# axis, the one triton vectors a load along, is the contiguous one -- was tried
# on the strength of a microbenchmark that read the isolated weight load 1.13 to
# 1.40x faster, and reverted.  On the whole call it is a regression: 19 cases
# averaged 0.244 against 0.268, with the FMA shapes losing worst ((2,4,16,16,16)
# 381 us to 2630 us, (8,3,224,224) 617 us to 9285 us) on weight tiles of a few
# dozen elements.  The tile is not what the kernel is waiting on; rewriting the
# addressing around it just moves the compiler's schedule.
#
# Loop nesting differs between the two kernels on purpose.  The 2D kernels are
# tap-major (static kh/kw outside, the channel loop inside); the 3D one is
# channel-major.  A runtime channel loop wrapped around 27 unrolled taps instead
# of 9 sends the Ascend compiler into a multi-minute unroll -- measured stuck
# for > 15 min on (1,16,4,4,4)/3x3x3.  Nine taps is under that limit; twenty-
# seven is not.
#
# For the same reason the 3D kernel's dot arm walks its taps in a *runtime*
# loop: 27 unrolled taps that each hold a tl.dot do not merely compile slowly,
# they hang the device.  On the 3D core case (2,16,16,16,16) bf16 with 16 input
# and 16 output channels -- every tile at tl.dot's minimum size -- a 100-launch
# loop never returned once in four attempts, while the identical loop on the FMA
# kernel finished in 0.5 s and the same 27-tap dot kernel at 1x3x3 (9 taps
# instead of 27) finished normally.  It is the tap count, not the tile size.
# The runtime loop is what makes the arm compile and run at all; see _DOT_3D for
# what it measures now that it is on.
# ---------------------------------------------------------------------------

# The spatial tile is the inner dimension, so it is chosen first: a whole output
# row when it fits, capped at _BLOCK_W_MAX registers' worth.  The channel tile
# then takes what is left of a fixed accumulator budget.  Both are picked per
# call because the kernel is launch- and mask-bound at the small end of the
# shape range: a fixed BLOCK_W of 256 over a 6-wide output masks 250 lanes away
# and leaves a single program for the whole device.
#
# BLOCK_W is never traded down to buy BLOCK_OC back, which is what this did
# until it was measured -- see _BLOCK_ELEMS.  BLOCK_W has a defect the budget
# does not show: every tap load is a run of BLOCK_W elements, so a wider tile
# buys longer runs and fewer programs at once, and the reduction tile is the one
# that can pay for it.
# Raised from 256 by measurement, because the axis it was trading against is not
# interchangeable with it.  512 is not a tuning preference either: on every 1-D
# shape in the suite it is the whole output row, and the row is what sets the
# length of a load.
_BLOCK_W_MAX = 512
# Ceiling on the output-channel tile.  This is the kernel's throughput knob,
# and the reason is in the loop's shape: one program loads a (BLOCK_C, BLOCK_W)
# input tile and reuses it for BLOCK_OC output channels, so the bytes it must
# pull per MAC are 2 / BLOCK_OC -- the channel tile cancels out of the load
# entirely and the *count of programs* is what BLOCK_OC divides.  The kernel is
# bound on the MTE2 pipe (aic_mte2_ratio 1.000 against a cube at 0.133), so that
# is the load it is bound on.  Measured on (8,256,64,64)/k3, whole call, bf16:
#
#   BLOCK_OC    16     32     64    128    256
#   us        7052   3441   1711    912    917
#
# i.e. exactly 1/BLOCK_OC up to 128 and flat after, so the ceiling is worth
# setting from the budget below rather than at a constant.  It used to be 32,
# which cost this shape 3.8x and capped every case at OC/32 times the loads it
# needed.
_BLOCK_OC_MAX = 256
# Target accumulator size, i.e. BLOCK_OC * BLOCK_W, in elements.  Raised twice
# by measurement, 4096 -> 8192 -> 32768, each time because BLOCK_OC was hitting
# it with output channels still left over.
#
# The last step is the one that matters, and it is measured on the device rather
# than by wall clock -- these calls are 30 us and the ~73 us host cost of a
# triton launch swamps them:
#
#   (8,256,64,64)/k3   (128, 64) 939 us   (256, 64) 622 us    1.51x
#   (32,64,512)/k3     (64, 128)  18.1 us (512, 64) 12.7 us   1.43x
#
# -- whole call, bf16, device time, where the pair is (BLOCK_OC, BLOCK_W).  Note
# what changed on the second shape: the winning tile is *wider*, not just
# bigger, because BLOCK_C gave up what BLOCK_W took (see _pick_block_c) and the
# budget is exactly conserved.  That is the whole rule here.
_BLOCK_ELEMS = 32768
# Channel tile for the tl.dot path, in elements.  A single dot over the whole
# channel count is what the sweep preferred (C_IN=64 in one dot 5.35 ms, split
# into 32s 8.50 ms, into 16s 12.86 ms), so this only has to cap how much of a
# very wide channel dim is reduced per dot.
_BLOCK_C_MAX = 64
# tl.dot needs every dimension at least this large; below it there is no dot.
_DOT_MIN = 16
# Ceiling on the tl.dot path's input tile (BLOCK_C * BLOCK_W), in elements.
#
# The Ascend unified buffer is 1572864 bits per vector unit.  Measured, the
# pipelined requirement is four times the tile's fp32 size: BiShengHIR asks for
# 2113792 bits at (BLOCK_C=64, BLOCK_W=256), and 4 * 16384 * 32 = 2097152, so
# the tile must not exceed 1572864 / 128 = 12288 elements -- which for power-of-
# two tiles means 8192.  Exceeding it is a correctness bound, not a speed knob:
# the worst shapes fail the build with "ub overflow", but a tile that only just
# overruns it, like the 16384-element one on (32,64,210,210)/k5/s1/p1, compiles
# and then faults the device's MTE at runtime.  Shrinking BLOCK_C rather than
# BLOCK_W is the cheaper half: measured 0.545 ms against 0.593 ms on
# (64,48,1024)/k5/s2.
_UB_TILE_MAX = 8192
# Tile for _compact_plane_kernel, swept at fixed volume on the (rows, 132) ->
# (rows, 130) shape: 64x256 161.8 us, 128x256 156.3, 256x256 172.9.  Every other
# form of the same copy is far worse -- see that kernel.
_COMPACT_BLOCK_W = 256
_COMPACT_BLOCK_R = 128


def _compact_block_w(ow):
    """Column tile for _compact_plane_kernel: the padded row, rounded up.

    ``_COMPACT_BLOCK_W`` alone is a fixed 256 columns whatever OW is, and the
    compaction's load is a masked ``(BLOCK_R, BLOCK_W)`` tile of a row that is
    only OW wide -- so at OW=32 it fetches eight times the bytes it keeps, and
    the pass costs more than the convolution it is compacting for: on
    (16,32,32,32)/k3/d2 the compaction is 31.7 us of an 81 us call against the
    conv kernel's own 24.9.  Rounding the tile down to OW costs nothing where OW
    is already past 256, and where it is not the tile becomes the row.
    """
    return min(_COMPACT_BLOCK_W, triton.next_power_of_2(max(1, ow)))


# Unrolled taps a kernel may have and still be handed fp16/bf16 tiles for its
# dots.  Not a tuning knob: see _arith_dtype.  9 is the same ceiling the 3D note
# records for unrolled taps that hold a dot at all, and it is the one the 2D
# kernel was measured against -- 3x3 keeps the native dtype and is 1.66x faster
# for it, 5x5 hangs the device.
_DOT_TAPS_MAX = 9
# Whether the 3-D kernel may take its tl.dot arm.  On, now that the arm's one
# bug is fixed.
#
# It was off because the arm was wrong, and it was wrong in one place: the
# weight tile's channel-reduction offset was missing the ``* w_c_stride`` that
# the 2-D kernel's dot arm carries, so on a (tap, C, OC) weight the tile walked
# the *output-channel* axis where it meant the channel axis.  Three taps of
# looking right is what the old note here described -- "the tap, channel and
# output addressing all read correctly against the 2-D kernel that shares them"
# -- and the error that note records is what the missing factor produces when
# the two axes are both 16 wide.  Measured against an fp64 CPU reference on the
# three shapes that reach the arm, every 3-D shape whose output tile clears
# tl.dot's minimum in all three dimensions:
#
#   (2,16,16,16,16)  k3 s1 g1   rel 1.392  ->  2.403e-03
#   (4,16,24,24,24)  k3 s1 g1   rel 1.397  ->  2.527e-03
#   (2,16,12,12,12)  k3 s1 g16  rel 1.399  ->  2.716e-03
#
# -- the FMA arm's own error is 2.5e-03 on these same shapes, so the arm is not
# merely closer, it is at the same noise floor as every other dot path here.
#
# What turning it on buys, whole call, bf16, gem device time in us:
#
#   [ 5] (2,16,16,16,16)  k3 s1    406  ->  217    1.9x
#   [19] (4,16,24,24,24)  k3 s1   1623  ->  759    2.1x
#   [20] (2,16,16,16,16)  k3 s2    206  ->  206    1.0x
#   [21] (2,16,12,12,12)  k3 g16   301  ->  177    1.7x
#   [22] (2, 4,16,16,16)  k3 s1    354  ->  354    1.0x
#
# The two that do not move are the two that never reach the arm: at stride 2 the
# output row is 8 wide, and at C_in 4 the contraction is 4 deep, both under
# tl.dot's minimum, so _can_use_dot keeps them on the FMA arm either way and
# neither pays anything for the arm existing.  Same reason the depthwise lift
# below stays switched on -- it keeps (2,16,12,12,12)/g16 off the BLOCK_OC=1 FMA
# path, and the arm it lifts into is this one, so a 3-D depthwise shape still
# clears tl.dot on the dense weight.
_DOT_3D = True
# Cubes per program.  Named rather than a literal at the launch site so it can
# be swept; see the note in _direct_conv2d.
_NUM_WARPS = 4
# Whether the 2-D non-depthwise, non-dilated shapes take the im2col+GEMM path
# (see _im2col_gemm_conv2d) instead of the direct per-tap kernel.  This is the
# "match CANN's default strategy" move: CANN runs these as a big-K GEMM with the
# window expansion done in hardware during the GM->L1 move (its Load3D), so the
# GEMM gets a contraction axis of KH*KW*C_in.  Triton cannot express Load3D, so
# the expansion here is a gather inside the GEMM's A-load -- the tap axis in
# ``iw = ow*SW + kw*DW`` is a per-lane vector-indexed load, and the ``k_off % C``
# / ``k_off // C`` on a *runtime* C are emulated divisions on a 256-wide vector.
# Both are exactly the two ops this backend charges ~4 orders of magnitude for
# (see the note at the top of this file and [[ascend-croslane-op-costs]]).
#
# Measured, whole call, bf16, gem device us, direct against im2col:
#
#   [ 1] (32, 64, 512)         32   ->   4555    142x
#   [ 3] (32, 64, 128, 128)  2020   -> 326899    162x
#   [ 6] (64, 48, 1024)       180   ->  21404    119x
#   [12] (32, 64, 210, 210)  8303   -> 687588     83x
#   [14] (8, 256, 64, 64)     618   ->1433895   2320x
#
# so the flag stays off.  CANN's default strategy does not transfer to triton
# here: the hardware move that makes im2col+GEMM win for CANN is Load3D, and
# without it the implicit gather costs far more than the direct kernel's
# redundant per-tap loads ever did.  (A materialised im2col -- strided copies
# into a (KH*KW*C, M) buffer then a clean GEMM -- is the only way to keep the
# big-K contraction without the gather, and it was not pursued: the matrix for
# the k5 s2 case is ~1.1 GB in bf16, a full write plus a full read that the
# direct kernel at 99.4% cube utilisation has no headroom to absorb.)
_USE_IM2COL_GEMM = False
# Reduction tile of the im2col GEMM, in whole input channels.  BLOCK_K =
# _GEMM_K_TAPS * next_power_of_2(C_in).  Larger = longer dots (fewer, cheaper
# flushes on the cube); swept on the whole call, not assumed.
_GEMM_K_TAPS = 1
# Walking several output tiles per program -- a runtime loop over the row-id
# axis, with the grid divided by the same factor -- was tried here and reverted,
# and the numbers are worth keeping because the microbenchmark says it should
# work.  On an isolated load loop shaped like this kernel's tiles, four tiles
# per program take MTE2 from 217 to 681 GB/s and eight to 878: the engine is
# paid for once per program, not once per byte, so a kernel with many small
# programs starves it however much traffic is in flight.
#
# In this kernel the loop loses.  Same session, whole call, device us, one
# against two and four tiles per program:
#
#   [ 4] (8,3,224,224)     682  ->  690  ->  719
#   [ 7] (16,24,2048)       61  ->   75  ->   68
#   [ 8] (8,8,8192) k11 s4  89  ->   89  ->  101
#   [13] (16,32,24,24) g2s2 322 ->  338  ->  360
#   [18] (16,32,32,32) asym 107 ->  128  ->  174
#
# -- twelve of fifteen 2D shapes worse at two tiles, and none better by more
# than the drift between runs.  What the microbenchmark does not have is the
# store and the tl.dot that end each tile: a loop holding those cannot be
# software-pipelined across iterations, while the hardware's own program
# scheduler was already overlapping those phases between programs.  The
# per-program cost is real, and it is not what this kernel waits on.
# Smallest width stride for which the split halo is built.  A module constant
# rather than a literal because the split is a fixed cost against a shrinking
# benefit and the crossover has to be re-measured per shape, not argued; see
# _pad_split_width for what it does and the note at the call site for what it
# is worth.  Two is the value the whole-call sweep there supports.
_SPLIT_MIN_STRIDE = 2

# Innermost halo run, in bytes, at which a strided ``copy_`` is still worth its
# one pass over the interior.  Above it the run is a contiguous line of at least
# one cache line, so ``out[..., PH:PH+H, PW:PW+W].copy_(input)`` moves every byte
# once at full bandwidth; below it the run degenerates into a per-element gather.
# The boundary is one cache line (128), which is where the measurements separate
# -- see _pad_input.
_PAD_RUN_BYTES = 128
# Elements each program of the flat-row halo kernel (_pad_flat_row_kernel)
# covers, used to derive its BLOCK_R from the row width.  It is a count and not
# a row count because the row width is what varies by 16x across the 1-D suite
# (512 to 8192): a fixed BLOCK_R would leave the widest rows with four programs
# and the narrowest with a thousand.
#
# Swept on the whole call over the five 1-D cases that reach it through the
# fused entry -- 4096 against 8192, medians of three, gems device time in us,
# torch against the same run so the ratio is what is being read:
#
#                     4096          8192
#   [ 1] (32,64,512)  21.9 / 1.810  21.6 / 1.919   rows 2048, W  512
#   [ 2] (16,32,1024) 25.1 / 1.819  21.5 / 2.181   rows  512, W 1024
#   [ 7] (16,24,2048) 55.7 / 1.110  49.3 / 1.259   rows  384, W 2048
#   [10] (16,32,512)  10.1 / 2.396  21.0 / 1.154   rows  512, W  512
#   [11] (8,4,512)     9.6 / 1.438   9.3 / 1.509   rows   32, W  512
#
# 4096 wins by +0.025 on the suite's mean, and it wins on [10] alone: that one
# case is 10.9 us better at 4096, and at a torque of 2/(22*10) it is worth more
# than the 0.3/3.6/6.4/0.3 us the other four give back.  That is the same [10]
# that picked 8192 before this kernel was fused, by the same kind of margin --
# at 8192 the pair of launches cost pad 3.2 + prep 3.2 us against 14.8 us for
# the fused one, and at 4096 the fused one is 5.4 us against that same pair.
# The optimum moved because the kernel did, and there is no shape law under it
# either way: [1] and [10] have the same W=512 and the same BLOCK_R here, and
# land on opposite sides of the crossover in *both* kernels.  Read the
# aggregate, and re-sweep this whenever the flat branch's kernel changes --
# which is the only reason the numbers above are worth writing down.
#
# The other two 1-D cases, [6] and [8], reach this constant through _pad_input
# rather than the fused entry (_direct_conv gives them the split-width halo, so
# their pad is the halo's own, un-fused launch).  They go the other way, 93.1
# against 85.0 and 32.4 against 31.9 at 4096, worth -0.003 between them.  The
# seven come to +0.022.
_PAD_FLAT_ELEMS = 4096
# Spare elements left at the end of every buffer the kernels index directly, so
# that a masked lane's address stays inside the allocation.  See _slack.
_SLACK_ELEMS = 8192
# Side of the square tile the two weight-stage kernels use: _prep_weight_kernel
# for both of its axes, _densify_kernel for the flat walk it does instead.  The
# weights here are at most a few hundred thousand elements, so this is sized for
# the load rather than the grid -- 64 elements is four 128-byte rows per
# program, which is the run length the MTE wants -- and it leaves the largest
# weight in the suite (256 x 256 x 3 x 3) a grid of 144 programs.
_PREP_BLOCK = 64
# Tiles for the fused split halo (_pad_split_cast_kernel): rows of the
# (n, c, hh) index space by columns of the split width axis.  Sized for the
# load rather than the grid -- one program reads BLOCK_ROWS * BLOCK_Q * SW
# contiguous input elements and writes the same volume as SW contiguous planes,
# so the tile wants to be a few KB.  ROWS must divide BLOCK_ROWS or the affine
# row ids overshoot the halo (there is no clamp to fall back on; see the
# kernel), and the deinterleave is issue-bound, so the row block is what moves
# the number: measured on (32,64,210,210)/k5/s2, 8 rows is 4.03 ms, 16 is 1.98,
# 32 is 1.05, 64 is 0.92.  Swept on the whole call, not assumed.
_SPLIT_BLOCK_ROWS = 64
_SPLIT_BLOCK_Q = 128
# Upper bound on BLOCK_ROWS * next_power_of_2(W) for _fused_split_cast_kernel,
# which loads a whole (unpadded) input row at once rather than tiling the width
# like _pad_split_cast_kernel.  A 64 x 512 fp32 tile already needs 3145728 bits
# against the 1572864-bit unified buffer (see the SW=4 note in
# _pad_split_cast), so 64 x 256 (16384 elements) is the last whole-row shape that
# fits; the wide 1-D cases ([8] W=8192, [75] W=1024) stay on the width-tiled
# kernel.  Measured, not assumed: 64 x 256 compiles, 64 x 512 does not.
_FUSE_SPLIT_MAX_TILE = 16384
# Width tile (input columns) for _fused_split_cast_1d_kernel, the 1-D width-tiled
# sibling of _fused_split_cast_kernel.  Same unified-buffer budget as the
# whole-row kernel -- BLOCK_ROWS x BLOCK_W must stay under ~16384 fp32 elements
# (64 x 256 here, the last shape that fits before the tile overflows the
# 1572864-bit buffer) -- so a wide 1-D row (W=1024) is cut into W/BLOCK_W tiles
# instead of overflowing like the whole-row load would.
_FUSE_SPLIT_BLOCK_W = 256


def _arith_dtype(input, use_dot, taps, runtime_taps=False):
    """The dtype the direct kernels should do their arithmetic in.

    fp32, except that a small-tap kernel taking the ``tl.dot`` branch can keep
    fp16/bf16.

    The upcast this file does for bf16 and fp16 buys nothing on the dot path.
    ``tl.dot``'s products are exact either way: the operands are already bf16 or
    fp16, so their product fits an fp32 accumulator exactly whether the cube was
    handed the native dtype or its fp32 image, and only the accumulation order
    differs.  What the upcast costs is real, though -- a pass over the input, a
    pass over the output, and half again the tile traffic feeding the cube --
    and it is what the benchmark's glue kernels are: on
    (32,64,128,128)/3x3/p2 the call is 3252 us, of which 2471 is the conv kernel
    and 203 is the input upcast plus output downcast.  Keeping the native dtype
    takes that case to 1962 us, 1.66x, and is the same product for the same
    accumulator bits.

    **``taps`` is not a heuristic.**  A bf16 dot does not merely compile slowly
    where an fp32 one is merely slow -- it hangs the *device*.  Same shape, same
    kernel, only the tile dtype changed: (32,64,210,210)/k5/s2/p1 is 25 unrolled
    taps over a C_IN=64 reduction, so 50 dots in the unrolled body, and it runs
    in 10 s from a cold cache in fp32 but never returns in bf16 -- 13 minutes of
    compiler CPU and then aicore timeout 507014, the device wedged.  It is the
    same wall the 3D note below describes, reached from a different side: 9
    unrolled taps is under it, 25 is over, and bf16 lowers what a tap costs
    enough to cross it at a tap count fp32 clears.

    The FMA branch has no such excuse and must stay fp32; the Ascend backend
    does not vectorize a bf16 load feeding a multiply, and the 1D shape that
    runs in 1.06 ms in fp32 takes 87.5 ms in bf16.

    There is no shortcut for the large-tap dot cases.  The obvious move -- walk
    the taps in a runtime loop (one dot in the unrolled body, so bf16 can no
    longer hang) and keep the native dtype -- was tried on the 3-D dot arm,
    which already uses exactly that loop, and it is a *regression*, not a save:
    the 3-D shapes sit at tl.dot's minimum tile size, where the cube is latency-
    bound rather than operand-bound, so bf16's smaller operands buy nothing and
    the runtime loop loses the load/dot pipeline the unrolled 2-D form gets.
    [5]/[19]/[21] went 212 -> 507 us, 751 -> 2272 us, 170 -> 385 us with the
    dot arm handed its native dtype.  See the note in _direct_conv3d.

    For the 2-D kernel the runtime loop is a different story, and ``runtime_taps``
    (the extra flag this function takes) is how the launcher asks for it.  The
    loop itself is a win or neutral in fp32 for the split-width large-kernel
    shapes -- staggering the tap loads instead of bursting 25 at once -- and the
    one thing it changes is whether the native dtype rides along.  On the
    *non-split* path that is a win (bf16 halves the load traffic and the tile is
    not cube-saturated): [18] (16,32,32,32)/3x5/s(2,1) goes 100 -> 78 us.  On the
    *split* path it is a loss, because the loop re-derives every tap address
    through the residue arithmetic the transposed width needs, and there the
    smaller operands no longer pay for the loop: [12] (32,64,210,210)/k5/s2 goes
    8324 -> 10378 us and [8] (8,8,8192)/k11/s4 goes 83 -> 120 us when the dot arm
    is handed its native dtype.  The launcher therefore passes
    ``runtime_taps`` only when ``not split_w``.
    """
    if (
        use_dot
        and (taps <= _DOT_TAPS_MAX or runtime_taps)
        and input.dtype in (torch.float16, torch.bfloat16)
    ):
        return input.dtype
    return torch.float32


@libentry()
@triton.jit
def _memset_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    # Zero the whole padded buffer in one flat, unmasked-on-the-interior pass.
    # This is the fill half of fill-then-copy: the halo and the tail are written
    # as zeros here, and the interior copy below overwrites the middle.  Splitting
    # the two is what keeps *every* load address affine -- no tap clamp, no halo
    # mask on the load -- which is the one thing the fused clamp variant could not
    # do without taking the interior load out of affine form (a 160x wall, see
    # _direct_conv2d_kernel's note).
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + off, 0.0, mask=off < numel)


@libentry()
@triton.jit
def _memset_prep_kernel(
    out_ptr,
    numel,
    w_ptr,
    wt_ptr,
    N_MEMSET: tl.constexpr,
    GRID1: tl.constexpr,
    S_OC: tl.constexpr,
    C: tl.constexpr,
    OC: tl.constexpr,
    T: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CT: tl.constexpr,
):
    """``_memset_kernel`` and ``_prep_weight_kernel`` in one launch.

    The two bodies are the kernels they replace, joined on the program id: the
    memset tiles take ``[0, N_MEMSET)`` and the weight tiles the rest.  Same join
    as _pad_flat_prep_kernel -- the branch must be a real one so the else-arm's
    ``pid - N_MEMSET`` stays non-negative, and ``GRID1`` is the prep kernel's
    second axis folded back out with a division by a constant -- applied here to
    the *split halo's* fill pass instead of the pad's.  The 1-D split halo
    already pays a full-buffer memset; this rides the weight permutation onto it
    so the prep's own launch (a fixed ~2.6 us) is saved.  The prep body is
    _prep_weight_kernel's unchanged, tn/cn naming and all.
    """
    pid = tl.program_id(0)
    if pid < N_MEMSET:
        off = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + off, 0.0, mask=off < numel)
    else:
        q = pid - N_MEMSET
        p0 = q // GRID1
        p1 = q - p0 * GRID1
        ct = p0 * BLOCK_CT + tl.arange(0, BLOCK_CT)
        oc = p1 * BLOCK_OC + tl.arange(0, BLOCK_OC)
        ct_ok = ct < C * T
        oc_ok = oc < OC
        cn = ct // T
        tn = ct - cn * T
        wv = tl.load(
            w_ptr + oc[:, None] * S_OC + ct[None, :],
            mask=oc_ok[:, None] & ct_ok[None, :],
            other=0.0,
        )
        if OUT_FP32:
            wv = wv.to(tl.float32)
        tl.store(
            wt_ptr + tn[:, None] * (C * OC) + cn[:, None] * OC + oc[None, :],
            tl.trans(wv),
            mask=ct_ok[:, None] & oc_ok[None, :],
        )


@libentry()
@triton.jit
def _pad_copy_interior_2d_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    PH,
    PW,
    in_plane,
    in_row,
    out_plane,
    out_row,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Copy the unpadded interior into its place in the padded buffer.  The plane
    # id is n*C (the two leading contiguous dims), and r/c come straight out of
    # the grid, so both the load (r, c) and the store (r+PH, c+PW) are affine in
    # the arange ids -- no clamp, and the only mask is the block overshoot.
    pid_plane = tl.program_id(0)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    m = (r < H)[:, None] & (c < W)[None, :]
    src = in_ptr + pid_plane * in_plane + r[:, None] * in_row + c[None, :]
    v = tl.load(src, mask=m, other=0.0)
    dst = (
        out_ptr
        + pid_plane * out_plane
        + (r[:, None] + PH) * out_row
        + (c[None, :] + PW)
    )
    tl.store(dst, v, mask=m)


@libentry()
@triton.jit
def _pad_flat_row_kernel(
    in_ptr,
    out_ptr,
    PW,
    Wp,
    W: tl.constexpr,
    BORDER: tl.constexpr,
    BLOCK_R: tl.constexpr,
    CAST_FP32: tl.constexpr,
):
    """Whole halo for an unpadded row axis, in one launch.

    Both halves of the pad are here: the interior is a row of ``W`` read from
    the source's own (contiguous) row stride and written one ``PW`` further on,
    and the two borders are masked stores of zero off the same ``BLOCK_R`` rows.
    ``_pad_input`` splits that into ``_memset_kernel`` over the whole padded
    buffer and ``_pad_copy_interior_2d_kernel`` over the interior, which is a
    second launch and a full extra write of the interior to zero a few columns
    that are not read as often as they are written.

    Every address here is affine and in bounds without a clamp.  The source is
    read at ``r*W + c`` with ``c < W`` exactly (the caller guarantees ``W`` is a
    power of two, so ``tl.arange`` can span it exactly -- this is the one place
    the power-of-two requirement buys something rather than costing it), and
    the destinations are ``r*Wp + c + PW`` for the interior plus the borders at
    ``r*Wp + b`` and ``r*Wp + W + PW + b``.  The right border's *masked* lanes
    reach ``BORDER - PW - tail`` past the end of the last row, which is why the
    allocation carries the usual slack: the mask suppresses the access, not the
    address, and the MTE faults on the address first.

    ``CAST_FP32`` reads a bf16/fp16 source and writes fp32 out.  The kernel
    already touches every element of the input, so an upcast here is a wider
    store and nothing else -- where ``input.to(arith)`` in front of it is a
    whole launch of its own (aclnnInplaceCopy_CastAiCore), 1.8 us on [11] and
    3.4 us on [7] against 9.4 and 53.1 us calls.  Only the interior run is cast;
    the two border stores write zeros, which are the same value either way.  See
    the ``cast`` gate in _pad_flat_row for when it is allowed.
    """
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, W)
    v = tl.load(in_ptr + r[:, None] * W + c[None, :])
    if CAST_FP32:
        v = v.to(tl.float32)
    tl.store(out_ptr + r[:, None] * Wp + (c[None, :] + PW), v)

    b = tl.arange(0, BORDER)
    rb = r[:, None] * Wp
    tl.store(out_ptr + rb + b[None, :], 0.0, mask=(b < PW)[None, :])
    tl.store(
        out_ptr + rb + (W + PW + b[None, :]),
        0.0,
        mask=(b < Wp - W - PW)[None, :],
    )


@libentry()
@triton.jit
def _pad_flat_prep_kernel(
    in_ptr,
    pad_ptr,
    w_ptr,
    wt_ptr,
    PW,
    Wp,
    N_PAD: tl.constexpr,
    GRID1: tl.constexpr,
    W: tl.constexpr,
    BORDER: tl.constexpr,
    BLOCK_R: tl.constexpr,
    S_OC: tl.constexpr,
    C: tl.constexpr,
    OC: tl.constexpr,
    T: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CT: tl.constexpr,
    CAST_FP32: tl.constexpr,
):
    """``_pad_flat_row_kernel`` and ``_prep_weight_kernel`` in one launch.

    The two bodies are exactly the kernels they replace, joined on the program
    id: the pad tiles take ``[0, N_PAD)`` and the weight tiles the rest.  The
    bodies share nothing, so this buys nothing but a launch -- which is the
    point.  A launch is 2.4 us of device time whatever the grid holds, and on
    the small 1-D shapes the whole call is three of them: (16,32,512)/k3 runs
    in 11.2 us and 1.5 us of that is the prep.  Nothing smaller can be fused
    out of these calls -- the pad has to finish before the conv reads it, and
    the conv is the third launch.

    The join is on ``tl.program_id``, and that has to be a real branch for the
    else-arm's ``pid - N_PAD`` to stay non-negative; if the backend ran both
    arms of every program it would index negatively and the MTE would fault on
    the address before any mask was consulted.  Checked directly on this
    backend rather than assumed -- a probe kernel with two disjoint stores and a
    grid that straddles N does fill both and only both.  ``N_PAD`` is constexpr
    so the test is against an immediate.

    ``GRID1`` is the prep kernel's second axis, folded in here because a
    single flat grid has one axis: the pad tiles are laid out first, so
    ``q = pid - N_PAD`` is the prep kernel's own ``p0 * GRID1 + p1`` and the two
    are recovered by a division by a constant.  Everything else about that body
    is unchanged, including why it is worth having (see _prep_weight).
    """
    pid = tl.program_id(0)
    if pid < N_PAD:
        r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.arange(0, W)
        v = tl.load(in_ptr + r[:, None] * W + c[None, :])
        if CAST_FP32:
            v = v.to(tl.float32)
        tl.store(pad_ptr + r[:, None] * Wp + (c[None, :] + PW), v)

        b = tl.arange(0, BORDER)
        rb = r[:, None] * Wp
        tl.store(pad_ptr + rb + b[None, :], 0.0, mask=(b < PW)[None, :])
        tl.store(
            pad_ptr + rb + (W + PW + b[None, :]),
            0.0,
            mask=(b < Wp - W - PW)[None, :],
        )
    else:
        q = pid - N_PAD
        p0 = q // GRID1
        p1 = q - p0 * GRID1
        ct = p0 * BLOCK_CT + tl.arange(0, BLOCK_CT)
        oc = p1 * BLOCK_OC + tl.arange(0, BLOCK_OC)
        ct_ok = ct < C * T
        oc_ok = oc < OC
        cn = ct // T
        tn = ct - cn * T
        # Named apart from the pad arm's ``v``, which it would otherwise join at
        # the merge point of the two branches: a name assigned on both sides has
        # to have one type, and these are (BLOCK_CT, BLOCK_OC) against
        # (BLOCK_R, W) whatever the element size.  Renamed, the two never meet.
        wv = tl.load(
            w_ptr + oc[:, None] * S_OC + ct[None, :],
            mask=oc_ok[:, None] & ct_ok[None, :],
            other=0.0,
        )
        if OUT_FP32:
            wv = wv.to(tl.float32)
        tl.store(
            wt_ptr + tn[:, None] * (C * OC) + cn[:, None] * OC + oc[None, :],
            tl.trans(wv),
            mask=ct_ok[:, None] & oc_ok[None, :],
        )


@libentry()
@triton.jit
def _pad_copy_interior_3d_kernel(
    in_ptr,
    out_ptr,
    D,
    H,
    W,
    PD,
    PH,
    PW,
    in_plane,
    in_d,
    in_h,
    out_plane,
    out_d,
    out_h,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Same interior copy as the 2-D kernel, but a 2-D (H, W) tile per depth
    # plane instead of one program per (d, h) row.  A one-row-per-program grid is
    # the anti-pattern _pad_split_width measures (~250 cycles to open a row
    # whatever its width) and the 3-D pad is what it hit before; d/h/w are all
    # plain grid/arange ids here, so every address stays affine.
    pid_plane = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    w = tl.arange(0, BLOCK_W)
    m = (h < H)[:, None] & (w < W)[None, :]
    src = in_ptr + pid_plane * in_plane + d * in_d + h[:, None] * in_h + w[None, :]
    v = tl.load(src, mask=m, other=0.0)
    dst = (
        out_ptr
        + pid_plane * out_plane
        + (d + PD) * out_d
        + (h[:, None] + PH) * out_h
        + (w[None, :] + PW)
    )
    tl.store(dst, v, mask=m)


@libentry()
@triton.jit
def _pad_zero_hband_kernel(
    out_ptr, row_base, Wp, out_plane, out_row, BLOCK_C: tl.constexpr
):
    # Zero a horizontal band of full-width rows starting at row_base (the top or
    # bottom padding).  The row is row_base + pr and the column is a flat run, so
    # the store address is affine and unmasked except for the column overshoot.
    pid = tl.program_id(0)
    pr = tl.program_id(1)
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(
        out_ptr + pid * out_plane + (row_base + pr) * out_row + c,
        0.0,
        mask=c < Wp,
    )


@libentry()
@triton.jit
def _pad_zero_vstrip_kernel(
    out_ptr,
    col_base,
    strip_w,
    H,
    PH,
    out_plane,
    out_row,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Zero a vertical strip of strip_w columns at col_base over the H interior
    # rows (offset PH).  r is a row block and p a column, both affine; the strip
    # is narrow so BLOCK_P is its next power of two.
    pid = tl.program_id(0)
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    p = tl.arange(0, BLOCK_P)
    m = (r < H)[:, None] & (p < strip_w)[None, :]
    tl.store(
        out_ptr
        + pid * out_plane
        + (r + PH)[:, None] * out_row
        + (col_base + p)[None, :],
        0.0,
        mask=m,
    )


def _pad_flat_row(input, padding, tail, weight=None, out_dtype=None):
    """The 1-D halo, optionally carrying the weight permutation in its launch.

    Returns ``(src, wt)``; ``(None, None)`` when the shape does not reach here,
    and ``(src, None)`` when it does but no weight was handed over.

    The 1-D lift, before either branch of ``_pad_input``, because both are the
    wrong shape for it.  ``_direct_conv`` lifts a 1-D call to 2-D with a leading
    unit *height*, so here H == 1 and PH == 0 -- which makes the padded planes
    contiguous in the flat row axis too (Hp == H, so out_plane == out_row ==
    Wp).  The (n, c, h) triple therefore collapses into a single row axis of
    N*C rows, and the interior copy becomes a plain 2-D strided copy with
    BLOCK_R rows per program.

    Left alone, a 1-D call takes the wide branch and aclnn's ViewCopy.  That
    copy is issued as N*C separate strided rows of 1 KB source against a
    destination stride of Wp*element_size -- 1028 B on (32,64,512)/k3, not a
    multiple of the cache line, so no two rows share an alignment.  It runs at
    153 GB/s: 26.8 us to move 4 MB, against a whole call of 50 us, and it is
    the largest single kernel on every 1-D shape in the suite (26.8 / 20.5 /
    15.8 / 14.1 / 7.6 / 7.3 us on [1] [2] [6] [7] [8] [11]).

    The row axis must divide BLOCK_R, which is what keeps every address in
    bounds: with rows % BLOCK_R == 0 no program overshoots on that axis, so the
    load needs neither a mask nor the allocator's slack (the two border stores
    do still carry the usual one; see the kernel).  W must be a power of two so
    ``tl.arange`` can span a whole row, which is the one place that requirement
    buys something rather than costing it.

    ``weight`` is the second entry point into the same launch: when it is given,
    the permutation ``_prep_weight`` would build in its own kernel is built here
    instead, and the launch with it is not spent.  Every 1-D case with a kernel
    larger than 1x1 arrives with one, which is all of them but [9]; the 1x1
    shapes have no permutation to fold in at all (see the gate in
    _direct_conv2d) and the caller passes None for them.  See
    _pad_flat_prep_kernel for the join.

    ``out_dtype`` is the arithmetic dtype, and it is also the dtype the halo is
    written in: when it differs from the input's the kernel does the upcast
    itself (see ``CAST_FP32``) and the launch ``input.to(arith)`` would have
    been is not spent.  It is the same argument that decides ``OUT_FP32`` for
    the weight, so the caller hands over ``arith`` either way and the cast
    happens wherever the copy happens.  The gate is ``is_contiguous``: the
    kernel addresses the source as ``r*W + c``, and today the only reason a
    strided input cannot reach it is that ``.to(arith)`` allocated a
    contiguous one on the way.  Skipping the cast means keeping the caller's
    tensor, so a strided one has to be turned away and _direct_conv2d sends it
    down the old path -- see the ``cast_in`` gate there for where that is
    decided.
    """
    # Before the shape test, and for the same reason _pad_input has it before
    # its branches: with no padding and no tail there is no halo to build, and
    # the input is already a valid source.  The launcher calls this directly
    # rather than through _pad_input, so it has to make that call itself --
    # leaving it to the fallback would find the flat branch unreachable anyway
    # (it would return the input unchanged) but only after a full copy of it,
    # which is what a 1x1 call with p=0 paid here: 1.5 us, one launch, for a
    # tensor the convolution reads as it stands.
    #
    # With a cast to do, that early-out is not free any more -- there is no
    # halo, but there is still the upcast, and it has to happen somewhere.  It
    # happens here, in the launch this function was supposed to be avoiding,
    # which is exactly what the caller would have paid without the fold.
    tail = max(0, tail)
    cast = out_dtype is not None and out_dtype != input.dtype
    if not any(padding) and tail == 0:
        return (input.to(out_dtype) if cast else input), None
    if not (input.ndim - 2 == 2 and input.shape[2] == 1 and padding[0] == 0):
        return None, None
    N, C, W = input.shape[0], input.shape[1], input.shape[3]
    PW = padding[1]
    rows = N * C
    Wp = W + 2 * PW + tail
    if W < 2 or W & (W - 1):
        return None, None
    block_r = max(1, _PAD_FLAT_ELEMS // W)
    while block_r > 1 and rows % block_r:
        block_r -= 1
    if rows % block_r or block_r * W > _UB_TILE_MAX:
        return None, None
    border = max(1, triton.next_power_of_2(PW + tail))
    buf = torch.empty(
        rows * Wp + max(_SLACK_ELEMS, border),
        device=input.device,
        dtype=out_dtype if cast else input.dtype,
    )
    out = buf[: rows * Wp].view(N, C, 1, Wp)
    if weight is None:
        _pad_flat_row_kernel[(rows // block_r,)](
            input,
            out,
            PW,
            Wp,
            W=W,
            BORDER=border,
            BLOCK_R=block_r,
            CAST_FP32=cast,
            num_warps=_NUM_WARPS,
        )
        return out, None
    oc, c = weight.shape[0], weight.shape[1]
    t = 1
    for size in weight.shape[2:]:
        t *= size
    out_dtype = weight.dtype if out_dtype is None else out_dtype
    numel = oc * c * t
    wbuf = torch.empty(numel + _SLACK_ELEMS, device=weight.device, dtype=out_dtype)
    wt = wbuf[:numel].view(*weight.shape[2:], c, oc)
    block_ct, block_oc = _prep_blocks(c * t, oc)
    grid1 = triton.cdiv(oc, block_oc)
    n_pad = rows // block_r
    _pad_flat_prep_kernel[(n_pad + triton.cdiv(c * t, block_ct) * grid1,)](
        input,
        out,
        weight.contiguous(),
        wt,
        PW,
        Wp,
        N_PAD=n_pad,
        GRID1=grid1,
        W=W,
        BORDER=border,
        BLOCK_R=block_r,
        S_OC=c * t,
        C=c,
        OC=oc,
        T=t,
        OUT_FP32=out_dtype == torch.float32,
        BLOCK_OC=block_oc,
        BLOCK_CT=block_ct,
        CAST_FP32=cast,
        num_warps=_NUM_WARPS,
    )
    return out, wt


def _pad_input(input, padding, tail):
    """Materialise the zero halo the kernels index through.

    The kernels use unpadded tap indices (``oh * SH + kh * DH``), so they need a
    tensor in which the halo already exists.  Building one costs a single pass
    over the input, and it buys the thing the alternatives cannot: the loads
    carry no mask, because a tap that falls in the halo reads a real zero --
    the padding is done by the data rather than by a compare per tap.

    That is not a stylistic choice.  The obvious way to keep a masked lane
    inside the tensor is a clamp, ``min(max(ow * SW + kw * DW - PW, 0), W - 1)``,
    and it takes the address out of the affine form the backend's axis analysis
    needs: the identical kernel then runs 160x slower -- 616 ms against 3.8 ms
    on (32,64,128,128)/k3/s2, at 97% cube utilisation either way.  See the note
    above _direct_conv2d_kernel.  Masking the load instead of clamping is closer
    -- it leaves the address alone -- but it is not free either: measured on the
    3-D shapes, on the FMA arm those shapes were running at the time, it costs
    the kernel more than the copy it saves ((2,16,16,16,16) at 226 us against
    943 us for the same kernel with ``w_ok`` on the loads).

    ``F.pad``'s two kernels are expensive for what they do and *flat* in size,
    which is why they dominate the small shapes:

        aclnnConstantPadNd_PadV3AiCore_MemSet    5.4 us
        aclnnConstantPadNd_PadV3AiCore_PadV3     7.7 us

    the same to within 0.1 us at 16 KB, 256 KB and 1 MB, against a triton
    kernel floor of 1.5 us -- while the whole call is around 30 us on the shapes
    where this matters, so the 13 us is nearly half of it.  Replacing the pair
    with triton kernels is the obvious move and it was tried twice; both were
    reverted, and the reasons are worth keeping:

    * fill then copy, two kernels: score 0.344 against the vendor pair's 0.386.
      The fill is a full pass over the *padded* buffer -- 1.4 GB on
      (32,64,210,210)/k5/s2 -- and that is the work the vendor pair does not do,
      because PadV3 writes every padded element once, interior and halo
      together.  Splitting the two makes the padding cost a pass of its own.
    * one fused kernel, one program per padded row: 0.332, and
      (32,64,128,128) went 2040 us to 16872 us.  That grid is 266240 programs
      each writing 130 columns, which is the per-row cost _pad_split_width
      measures below -- roughly 250 cycles to open a row whatever its width --
      so one row per program is the worst shape this can take.  Giving each
      program a 2-D block instead means a column range that starts at
      ``o - PW`` for the input tile, negative on the leftmost tile, and a
      masked lane still forms an address (see _slack).  That is the same wall
      the halo exists to route around, one level down.

    * copy and halo, two kernels, interior and border each in its own pass, with
      the halo run as a 3-D grid so that no index comes out of a flat division:
      score 0.367 against the vendor pair's 0.408.  It is correct -- all twelve
      shapes bit-exact against ``F.pad``, 22/22 cases -- and it is slower for a
      reason no tiling fixes.  On (32,64,512)/k3 the zeroing kernel is 64.1 us
      and the copy 24.8 us against 15.1 us for both vendor kernels together: the
      zeroing kernel stores into 4096 halo cells out of 1.57 M lanes and a masked
      store issues one lane at a time here, and the copy writes its interior as
      1 KB rows at a 2056 B stride for 242 GB/s where PadV3 does the same strided
      work *and* the border in 15.1 us.

    ``tail`` is how far past the end of a padded row the *last* output tile's
    lanes run.  Those lanes are masked at the store and their columns are
    thrown away, but they still form an address, and a masked address is still
    an address: the MTE faults on one outside the allocation before the mask is
    consulted (see _slack).  It costs nothing to give them room *inside* the
    allocation instead, by widening the padded row on the right -- the extra
    columns are zeros and no unmasked lane ever reads them.

    Callers hand this an fp32 tensor even for fp16/bf16 inputs, so the halo is
    fp32 like the accumulator.
    """
    tail = max(0, tail)
    if not any(padding) and tail == 0:
        return input
    # The 1-D lift, before either branch below, because both are the wrong shape
    # for it.  Shapes it does not take fall through unchanged.
    flat = _pad_flat_row(input, padding, tail)
    if flat[0] is not None:
        return flat[0]
    # Wide innermost run: the halo is a ``torch.zeros`` + strided ``copy_``.  The
    # copy writes the interior through a view whose innermost axis is a full
    # contiguous line (>= one cache line), so it moves every byte once at ~420
    # GB/s -- 235 us on (32,64,128,128)/p2 -- and the memset is already the cheap
    # pass.  A triton memset + interior-copy pair is slower for the same interior
    # (the two triton fill/copy and copy/halo formulations were measured and
    # reverted; see the note above), so only the narrow runs below take triton.
    if input.ndim - 2 == 2 and input.shape[-1] * input.element_size() >= _PAD_RUN_BYTES:
        N, C, H, W = input.shape
        PH, PW = padding
        Hp = H + 2 * PH
        Wp = W + 2 * PW + tail
        out = torch.zeros((N, C, Hp, Wp), device=input.device, dtype=input.dtype)
        out[:, :, PH : PH + H, PW : PW + W].copy_(input)
        return out
    # Narrow 2-D runs and every 3-D shape: the strided copy_ degenerates into a
    # per-element gather there, so self-implement the halo as a flat memset plus
    # an affine interior copy.  Both passes keep every address affine -- the fill
    # is a pure flat store and the copy's load/store are plain arange ids offset
    # by the padding constants -- because the one fused single-pass formulation
    # must clamp the halo lanes' source address, and that clamp is the 160x wall
    # _direct_conv2d_kernel documents.
    if input.ndim - 2 == 2:
        N, C, H, W = input.shape
        PH, PW = padding
        Hp = H + 2 * PH
        Wp = W + 2 * PW + tail
        out = torch.empty((N, C, Hp, Wp), device=input.device, dtype=input.dtype)
        numel = N * C * Hp * Wp
        BLOCK = 1024
        _memset_kernel[(triton.cdiv(numel, BLOCK),)](
            out, numel, BLOCK=BLOCK, num_warps=_NUM_WARPS
        )
        BLOCK_R, BLOCK_C = 64, 128
        grid = (N * C, triton.cdiv(H, BLOCK_R), triton.cdiv(W, BLOCK_C))
        _pad_copy_interior_2d_kernel[grid](
            input,
            out,
            H,
            W,
            PH,
            PW,
            H * W,
            W,
            Hp * Wp,
            Wp,
            BLOCK_R=BLOCK_R,
            BLOCK_C=BLOCK_C,
            num_warps=_NUM_WARPS,
        )
        return out
    N, C, D, H, W = input.shape
    PD, PH, PW = padding
    Dp = D + 2 * PD
    Hp = H + 2 * PH
    Wp = W + 2 * PW + tail
    out = torch.empty((N, C, Dp, Hp, Wp), device=input.device, dtype=input.dtype)
    numel = N * C * Dp * Hp * Wp
    BLOCK = 1024
    _memset_kernel[(triton.cdiv(numel, BLOCK),)](
        out, numel, BLOCK=BLOCK, num_warps=_NUM_WARPS
    )
    BLOCK_H = min(32, triton.next_power_of_2(H))
    BLOCK_W = min(32, triton.next_power_of_2(W))
    grid = (N * C, D, triton.cdiv(H, BLOCK_H))
    _pad_copy_interior_3d_kernel[grid](
        input,
        out,
        D,
        H,
        W,
        PD,
        PH,
        PW,
        D * H * W,
        H * W,
        W,
        Dp * Hp * Wp,
        Hp * Wp,
        Wp,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=_NUM_WARPS,
    )
    return out


def _pad_split_width(input, padding, sw, dw, kw, out_w, block_w):
    """The halo of _pad_input, with the width axis split into ``sw`` planes.

    A tap for output column ``ow`` reads the input at ``ow * SW + kw * DW``,
    which for ``SW > 1`` is a stride-SW run.  The MTE pays for the cache lines
    a run touches rather than the bytes it keeps, so that costs ``SW`` times the
    traffic, and past stride 2 the backend stops treating it as a run at all.
    Measured through this entry point on (8,8,8192)/k11/p5, same taps and same
    input throughout, only the width stride varying:

        stride 1   OW 8192    422 us    0.052 us per output column
        stride 2   OW 4096    422 us    0.103 us per output column
        stride 4   OW 2048   1532 us    0.748 us per output column

    -- exactly 2x at stride 2, as the line count predicts, and 14.5x at stride
    4, which is the run degenerating.

    Reshaping the padded width to ``(W/SW, SW)`` and transposing puts element
    ``q * SW + r`` at ``[r, q]``, so the same tap becomes plane
    ``(kw*DW) % SW`` at column ``ow + (kw*DW)//SW``: unit stride in ``ow``, with
    the plane a launch-time constant.  No element moves and no arithmetic
    changes, only the order they sit in, so the result is bit-identical.

    The halo itself is still built by _pad_input, in one F.pad; this adds one
    transposing copy on top of it.
    """
    # No tail: the split layout has its own, appended flat past the last plane,
    # because the overshoot there leaves the row rather than the tensor.
    # F.pad already returns a contiguous tensor; the no-padding case returns the
    # operand itself, which the view below needs flattened first.
    halo = _pad_input(input, padding, 0).contiguous()
    n, c = halo.shape[0], halo.shape[1]
    spatial = halo.shape[2:-1]
    wq = triton.cdiv(halo.shape[-1], sw)
    if wq * sw != halo.shape[-1]:
        # _pad_input pads symmetrically, so the split can need up to SW-1
        # columns of zeros on the right to make the planes whole.
        halo = _pad_input(halo, (0,) * (halo.ndim - 2), wq * sw - halo.shape[-1])
    # The flat layout lets a masked lane of the last output tile overshoot past
    # the end of its row, and _pad_input gives it room by widening the padded
    # row.  Splitting the width removes that room -- the row's overshoot is now
    # the plane's -- so the same allowance has to be appended past the last
    # plane instead: the highest address a tap can form is (sw-1)*wq + ow_max,
    # against a tensor of sw*wq elements.
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = n * c * sw * wq
    for s in spatial:
        numel *= s
    # Not zeros: the transposing copy below writes every element of ``out``, so
    # a zero fill here is a pass over the whole tensor whose output is
    # immediately overwritten.  The tail past ``numel`` is only ever addressed
    # by lanes the kernel masks off, so it needs room rather than a value.
    buf = torch.empty(
        numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=input.dtype
    )
    out = buf[:numel].view(n, c, *spatial, sw, wq)
    out.copy_(halo.view(n, c, *spatial, wq, sw).transpose(-1, -2))
    return out


@libentry()
@triton.jit
def _pad_split_cast_kernel(
    halo_ptr,
    out_ptr,
    ROWS,
    WQ,
    halo_row_stride,
    out_row_stride,
    out_r_stride,
    SW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
):
    """Split (SW=2 or 4) and optionally upcast a pre-padded halo in one pass.

    Replaces the three-kernel chain that built the split halo until now --
    ``input.to(fp32)`` (a Cast), ``_pad_input``'s F.pad (PadV3), and the
    ``copy_(view(...).transpose(-1, -2))`` deinterleave (a vendor Transpose).
    All three are issue-bound on their access pattern rather than bandwidth:
    measured on (32,64,210,210)/k5/s2, the Transpose runs a (Wq, SW) plane with
    SW=2 through a 2-wide transpose at ~50 GB/s where a flat copy does 400, and
    it is 3467 us of an 8297 us call -- the largest single kernel, more than the
    convolution itself.

    The halo is read as a flat (ROWS, halo_row_stride) tensor with plain
    ``tl.arange`` row and column ids and no mask or clamp.  That is a hard
    requirement of this backend: ``reshape`` + ``permute`` silently produce zero
    output whenever the tensor they fold came from a load whose index passed
    through any non-affine expression, a ``tl.minimum`` clamp even when it is a
    no-op.  _pad_split_cast widens the halo and requires ROWS to divide
    BLOCK_ROWS so these ids are always in-bounds and no clamp is needed.

    The deinterleave is ``tl.split``, not ``reshape`` + ``permute``: the latter
    is correct only on the affine load but lowers, measured here, to a 66 ms pass
    on (32,64,210,210)/k5/s2 where the same (SW, Wq) reorder through
    ``tl.split`` is 0.92 ms at BLOCK_ROWS=64.

    SW=4 nests two 2-way ``tl.split``s: folding the row into (Q, 2, 2) puts the
    column residue as ``c = 4q + 2a + b``, so the outer split on ``b`` separates
    even/odd and the inner split on ``a`` separates every-other, yielding the
    four planes (0, 2) and (1, 3).  Stride 4 is the only other width stride the
    benchmark carries ([8] (8,8,8192)/k11/s4), and the generic transposing copy
    it used to take is 45 us of a 72 us call.
    """
    pid_row = tl.program_id(0)
    pid_q = tl.program_id(1)

    rows = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    j = pid_q * (BLOCK_Q * SW) + tl.arange(0, BLOCK_Q * SW)
    v = tl.load(halo_ptr + rows[:, None] * halo_row_stride + j[None, :])
    if CAST_TO_FP32:
        v = v.to(tl.float32)

    out_base = out_ptr + rows[:, None] * out_row_stride + q[None, :]
    m = (rows < ROWS)[:, None] & (q < WQ)[None, :]
    if SW == 2:
        # v is (BLOCK_ROWS, BLOCK_Q*2); folding the plane axis out and splitting
        # it is the (SW=2, WQ) reorder -- the even column of every run becomes
        # the r=0 plane and the odd one r=1.
        v = tl.reshape(v, (BLOCK_ROWS, BLOCK_Q, 2))
        even, odd = tl.split(v)
        tl.store(out_base + 0 * out_r_stride, even, mask=m)
        tl.store(out_base + 1 * out_r_stride, odd, mask=m)
    else:
        v = tl.reshape(v, (BLOCK_ROWS, BLOCK_Q, 2, 2))
        lo, hi = tl.split(v)
        p0, p2 = tl.split(lo)
        p1, p3 = tl.split(hi)
        tl.store(out_base + 0 * out_r_stride, p0, mask=m)
        tl.store(out_base + 1 * out_r_stride, p1, mask=m)
        tl.store(out_base + 2 * out_r_stride, p2, mask=m)
        tl.store(out_base + 3 * out_r_stride, p3, mask=m)


@libentry()
@triton.jit
def _fused_split_cast_kernel(
    in_ptr,
    out_ptr,
    H,
    W,
    PH,
    PW,
    in_plane_stride,
    out_plane_stride,
    out_row_stride,
    out_r_stride,
    WQ: tl.constexpr,
    SW: tl.constexpr,
    W_POW2: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
):
    """Split (SW=2/4) + pad + upcast from the *unpadded* input in one pass.

    This is _pad_split_cast_kernel with the halo materialisation folded away: it
    reads the raw (contiguous) input directly, folds the padding shift into the
    split-store (plane ``(s+PW)%SW`` at ``q+(s+PW)//SW``), and leaves every border
    -- top/bottom rows, left/right columns -- to the memset that precedes it in
    _fused_split_cast.  The load index ``r*W + c`` is affine; the only mask is
    the block/width overshoot expressed as a *value* mask (other=0), which
    tl.split carries correctly here, unlike a clamped/non-affine index (see
    _pad_split_cast_kernel's note).  This removes the zeros+copy_ halo build
    (1047 us of the (32,64,210,210)/k5/s2 call) at the price of a value-masked
    load; measured, the net is ~0.76x on that call's split prep.
    """
    pid_plane = tl.program_id(0)
    pid_row = tl.program_id(1)
    r = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.arange(0, W_POW2)
    rm = r < H
    v = tl.load(
        in_ptr + pid_plane * in_plane_stride + r[:, None] * W + c[None, :],
        mask=rm[:, None] & (c < W)[None, :],
        other=0.0,
    )
    if CAST_TO_FP32:
        v = v.to(tl.float32)
    q = tl.arange(0, W_POW2 // SW)
    base = out_ptr + pid_plane * out_plane_stride + (r + PH) * out_row_stride
    if SW == 2:
        v = tl.reshape(v, (BLOCK_ROWS, W_POW2 // 2, 2))
        even, odd = tl.split(v)
        e_pl = (0 + PW) % 2
        e_sh = (0 + PW) // 2
        o_pl = (1 + PW) % 2
        o_sh = (1 + PW) // 2
        tl.store(
            base[:, None] + e_pl * out_r_stride + q[None, :] + e_sh,
            even,
            mask=rm[:, None] & (q[None, :] + e_sh < WQ),
        )
        tl.store(
            base[:, None] + o_pl * out_r_stride + q[None, :] + o_sh,
            odd,
            mask=rm[:, None] & (q[None, :] + o_sh < WQ),
        )
    else:
        v = tl.reshape(v, (BLOCK_ROWS, W_POW2 // 4, 2, 2))
        lo, hi = tl.split(v)
        p0, p2 = tl.split(lo)
        p1, p3 = tl.split(hi)
        p0_pl = (0 + PW) % 4
        p0_sh = (0 + PW) // 4
        p1_pl = (1 + PW) % 4
        p1_sh = (1 + PW) // 4
        p2_pl = (2 + PW) % 4
        p2_sh = (2 + PW) // 4
        p3_pl = (3 + PW) % 4
        p3_sh = (3 + PW) // 4
        tl.store(
            base[:, None] + p0_pl * out_r_stride + q[None, :] + p0_sh,
            p0,
            mask=rm[:, None] & (q[None, :] + p0_sh < WQ),
        )
        tl.store(
            base[:, None] + p1_pl * out_r_stride + q[None, :] + p1_sh,
            p1,
            mask=rm[:, None] & (q[None, :] + p1_sh < WQ),
        )
        tl.store(
            base[:, None] + p2_pl * out_r_stride + q[None, :] + p2_sh,
            p2,
            mask=rm[:, None] & (q[None, :] + p2_sh < WQ),
        )
        tl.store(
            base[:, None] + p3_pl * out_r_stride + q[None, :] + p3_sh,
            p3,
            mask=rm[:, None] & (q[None, :] + p3_sh < WQ),
        )


@libentry()
@triton.jit
def _fused_split_cast_1d_kernel(
    in_ptr,
    out_ptr,
    W,
    PW,
    in_row_stride,
    out_plane_stride,
    out_r_stride,
    WQ: tl.constexpr,
    SW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    CAST_TO_FP32: tl.constexpr,
):
    """Width-tiled split + pad + upcast from the unpadded 1-D input.

    _fused_split_cast_kernel reads a *whole* input row, which overflows the
    unified buffer for wide 1-D rows (W=1024 x BLOCK_ROWS=64 fp32 = 2 Mbit).
    This is the same fold, cut along the width: each program takes BLOCK_W
    input columns, so the (BLOCK_ROWS, BLOCK_W) fp32 tile stays under the
    budget.  The 1-D lift gives H == 1 and PH == 0, so the (n, c, h) triple
    collapses into one flat row axis of N*C rows (the launcher requires
    W % BLOCK_W == 0 and N*C % BLOCK_ROWS == 0, so the load is mask-free like
    _pad_split_cast_kernel's, not value-masked like _fused_split_cast_kernel's).

    The pad is folded into the store exactly as in _fused_split_cast_kernel,
    only the column origin is the tile's ``c0 = pid_w * BLOCK_W`` rather than 0:
    sub-plane ``s`` (column residue c0 + s mod SW) lands at plane
    ``(c0+s+PW) % SW``, q-offset ``(c0+s+PW) // SW``.  Both are scalar broadcast
    offsets onto an affine ``q`` index, so the store stays affine (the +1 shift
    on the odd sub-plane is free, not a gather).
    """
    pid_row = tl.program_id(0)
    pid_w = tl.program_id(1)
    r = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    v = tl.load(in_ptr + r[:, None] * in_row_stride + c[None, :])
    if CAST_TO_FP32:
        v = v.to(tl.float32)
    q = tl.arange(0, BLOCK_W // SW)
    base = out_ptr + r[:, None] * out_plane_stride
    c0 = pid_w * BLOCK_W
    if SW == 2:
        v = tl.reshape(v, (BLOCK_ROWS, BLOCK_W // 2, 2))
        even, odd = tl.split(v)
        e_pl = (c0 + 0 + PW) % 2
        e_sh = (c0 + 0 + PW) // 2
        o_pl = (c0 + 1 + PW) % 2
        o_sh = (c0 + 1 + PW) // 2
        tl.store(
            base + e_pl * out_r_stride + q[None, :] + e_sh,
            even,
            mask=(q[None, :] + e_sh < WQ),
        )
        tl.store(
            base + o_pl * out_r_stride + q[None, :] + o_sh,
            odd,
            mask=(q[None, :] + o_sh < WQ),
        )
    else:
        v = tl.reshape(v, (BLOCK_ROWS, BLOCK_W // 4, 2, 2))
        lo, hi = tl.split(v)
        p0, p2 = tl.split(lo)
        p1, p3 = tl.split(hi)
        p0_pl = (c0 + 0 + PW) % 4
        p0_sh = (c0 + 0 + PW) // 4
        p1_pl = (c0 + 1 + PW) % 4
        p1_sh = (c0 + 1 + PW) // 4
        p2_pl = (c0 + 2 + PW) % 4
        p2_sh = (c0 + 2 + PW) // 4
        p3_pl = (c0 + 3 + PW) % 4
        p3_sh = (c0 + 3 + PW) // 4
        tl.store(
            base + p0_pl * out_r_stride + q[None, :] + p0_sh,
            p0,
            mask=(q[None, :] + p0_sh < WQ),
        )
        tl.store(
            base + p1_pl * out_r_stride + q[None, :] + p1_sh,
            p1,
            mask=(q[None, :] + p1_sh < WQ),
        )
        tl.store(
            base + p2_pl * out_r_stride + q[None, :] + p2_sh,
            p2,
            mask=(q[None, :] + p2_sh < WQ),
        )
        tl.store(
            base + p3_pl * out_r_stride + q[None, :] + p3_sh,
            p3,
            mask=(q[None, :] + p3_sh < WQ),
        )


def _fused_split_cast_1d(
    input, padding, sw, dw, kw, out_w, block_w, arith, weight=None
):
    """The split halo for a 1-D stride-2/4 shape, built from the unpadded input.

    Same zero-then-overwrite shape as _fused_split_cast, specialised to the 1-D
    lift (H == 1, PH == 0): the row axis is just N*C, the input is a flat
    (N*C, W) run, and the left/right padding q-columns the split-store leaves
    behind are zeroed by the flat memset that precedes the kernel.  The width is
    tiled by BLOCK_W so a wide row (W=1024) does not overflow the unified
    buffer, which is exactly the case _fused_split_cast's whole-row load cannot
    take.

    ``weight``, when given, has its ``_prep_weight`` permutation folded onto the
    memset fill (see _memset_prep_kernel), so the call returns ``(out, wt)`` and
    the caller skips the prep's own launch.  When it is None the memset is the
    plain pass and the return is ``(out, None)``.
    """
    N, C, H, W = input.shape
    PH, PW = padding
    Wp = W + 2 * PW
    wq = triton.cdiv(Wp, sw)
    rows = N * C
    block_rows = _SPLIT_BLOCK_ROWS
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = rows * sw * wq
    buf = torch.empty(numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=arith)
    # Full-buffer memset, as in _fused_split_cast.  The border is SW thin strips
    # strided by wq (q < PW//SW and q >= wq-(W-1+PW)//SW-1), and a strip-only
    # zero is a strided single-column store -- the backend issues it one lane at
    # a time, measured slower (7.6 us) than zeroing the whole 12 MB contiguously
    # (6.2 us).  The interior is overwritten by the split kernel; zeroing it
    # first is the cheap pass that makes the strided border store unnecessary.
    wt = None
    if weight is not None:
        oc, c = weight.shape[0], weight.shape[1]
        t = 1
        for size in weight.shape[2:]:
            t *= size
        wt_numel = oc * c * t
        wt_buf = torch.empty(wt_numel + _SLACK_ELEMS, device=weight.device, dtype=arith)
        wt = wt_buf[:wt_numel].view(*weight.shape[2:], c, oc)
        block_ct, block_oc = _prep_blocks(c * t, oc)
        n_memset = triton.cdiv(buf.numel(), 4096)
        grid1 = triton.cdiv(oc, block_oc)
        _memset_prep_kernel[(n_memset + triton.cdiv(c * t, block_ct) * grid1,)](
            buf,
            buf.numel(),
            weight.contiguous(),
            wt,
            N_MEMSET=n_memset,
            GRID1=grid1,
            S_OC=c * t,
            C=c,
            OC=oc,
            T=t,
            OUT_FP32=(arith == torch.float32),
            BLOCK=4096,
            BLOCK_OC=block_oc,
            BLOCK_CT=block_ct,
            num_warps=_NUM_WARPS,
        )
    else:
        _memset_kernel[(triton.cdiv(buf.numel(), 4096),)](
            buf, buf.numel(), BLOCK=4096, num_warps=_NUM_WARPS
        )
    out = buf[:numel].view(N, C, H, sw, wq)
    grid = (
        triton.cdiv(rows, block_rows),
        triton.cdiv(W, _FUSE_SPLIT_BLOCK_W),
    )
    _fused_split_cast_1d_kernel[grid](
        input,
        out,
        W,
        PW,
        H * W,  # in_row_stride
        sw * wq,  # out_plane_stride (Hp == H == 1)
        wq,  # out_r_stride
        WQ=wq,
        SW=sw,
        BLOCK_ROWS=block_rows,
        BLOCK_W=_FUSE_SPLIT_BLOCK_W,
        CAST_TO_FP32=(arith == torch.float32 and input.dtype != torch.float32),
        num_warps=_NUM_WARPS,
    )
    return out, wt


def _fused_split_cast(input, padding, sw, dw, kw, out_w, block_w, arith):
    """The split halo built from the unpadded input, padding folded into the split.

    Zero-fill the whole split buffer (borders + tail) with one flat memset, then
    _fused_split_cast_kernel overwrites the interior straight from ``input``.
    The zero-then-overwrite is the same fill-then-copy shape the wide branch of
    _pad_input rejects *for the halo* (its fill is 1.4 GB); here the fill is the
    split buffer itself and the copy is replaced by the split, so the pass the
    vendor pair does not do is the one being eliminated, not added.  SW=2 and
    SW=4 with any PW < 2*SW are handled by the same plane-rotation rule.
    """
    N, C, H, W = input.shape
    PH, PW = padding
    Hp = H + 2 * PH
    Wp = W + 2 * PW
    wq = triton.cdiv(Wp, sw)
    block_rows = _SPLIT_BLOCK_ROWS // 2 if sw == 4 else _SPLIT_BLOCK_ROWS
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = N * C * Hp * sw * wq
    buf = torch.empty(numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=arith)
    # BLOCK=4096 holds the memset at its ~1.5 TB/s ceiling (BLOCK=1024 is 2.5x
    # slower); the fill is the one pass this path adds over the vendor pair.
    _memset_kernel[(triton.cdiv(buf.numel(), 4096),)](
        buf, buf.numel(), BLOCK=4096, num_warps=_NUM_WARPS
    )
    out = buf[:numel].view(N, C, Hp, sw, wq)
    W_POW2 = triton.next_power_of_2(W)
    grid = (N * C, triton.cdiv(H, block_rows))
    _fused_split_cast_kernel[grid](
        input,
        out,
        H,
        W,
        PH,
        PW,
        H * W,
        Hp * sw * wq,
        sw * wq,
        wq,
        WQ=wq,
        SW=sw,
        W_POW2=W_POW2,
        BLOCK_ROWS=block_rows,
        CAST_TO_FP32=(arith == torch.float32 and input.dtype != torch.float32),
        num_warps=_NUM_WARPS,
    )
    return out


def _pad_split_cast(input, padding, sw, dw, kw, out_w, block_w, arith, weight=None):
    """Build the split halo from a pre-padded halo, split + upcast in one pass.

    The halo is built by ``_pad_input`` (its wide branch is ``torch.zeros`` +
    ``copy_`` -- a plain memset + memcpy, no arithmetic), then
    ``_pad_split_cast_kernel`` reads it mask-free and does the (SW, WQ)
    deinterleave with ``tl.split`` while upcasting bf16/fp16 to ``arith``.  The
    result is bit-identical to ``_pad_split_width(input.to(arith), ...)`` -- same
    zeros, same cast, same (SW, WQ) layout.  SW=2 and SW=4 are the two width
    strides the benchmark carries; any other stride falls back to _pad_split_width.

    Returns ``(src, wt)``.  ``wt`` is None unless the 1-D fused path both fires
    and was handed a ``weight`` to permute; ``_direct_conv2d`` threads that
    weight through so the prep's own launch is skipped on that path.
    """
    N, C, H, W = input.shape
    if sw not in (2, 4):
        return (
            _pad_split_width(input.to(arith), padding, sw, dw, kw, out_w, block_w),
            None,
        )
    PH, PW = padding
    Hp = H + 2 * PH
    Wp = W + 2 * PW
    wq = triton.cdiv(Wp, sw)
    rows = N * C * Hp

    # The 4-way split reads SW=4 times as many columns per row, so its (rows,
    # cols) register tile is 2x the 2-way one and must shrink to stay inside the
    # unified-buffer budget (see _UB_TILE_MAX): 64 x 512 fp32 needs 3145728 bits
    # where the buffer is 1572864.  Halving both axes keeps the tile under it.
    block_rows = _SPLIT_BLOCK_ROWS // 2 if sw == 4 else _SPLIT_BLOCK_ROWS
    block_q = _SPLIT_BLOCK_Q // 2 if sw == 4 else _SPLIT_BLOCK_Q

    # Fused path: build the split halo straight from the unpadded input, folding
    # the pad into the split-store.  This drops the zeros + copy_ halo
    # materialisation (the largest glue kernel on the stride-2 cases) but needs a
    # contiguous input and a whole-row load that fits the unified buffer, so the
    # wide 1-D shapes stay on the width-tiled _pad_split_cast_kernel below.
    if (
        sw == 2
        and input.is_contiguous()
        and triton.next_power_of_2(W) * block_rows <= _FUSE_SPLIT_MAX_TILE
    ):
        return (
            _fused_split_cast(input, padding, sw, dw, kw, out_w, block_w, arith),
            None,
        )

    # 1-D width-tiled fused path: the whole-row load above overflows the unified
    # buffer for a wide 1-D row (W=1024), so tile the width instead.  Only the
    # 1-D lift reaches here -- H == 1 and PH == 0 collapse (n, c, h) into N*C
    # flat rows -- and it needs W a power of two and N*C % BLOCK_ROWS == 0 so the
    # load stays mask-free (see _fused_split_cast_1d_kernel).  Stride 4 is held
    # back because the only such case, [8], is already >0.85 and the width-tiled
    # _pad_split_cast_kernel below is not its bottleneck.
    if (
        sw == 2
        and H == 1
        and PH == 0
        and input.is_contiguous()
        and W > 1
        and W & (W - 1) == 0
        and rows % block_rows == 0
    ):
        return _fused_split_cast_1d(
            input, padding, sw, dw, kw, out_w, block_w, arith, weight
        )

    # The furthest column a conv tap can address in the split layout, for the
    # buffer's masked-lane allowance (see _slack); same as _pad_split_width.
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = rows * sw * wq
    buf = torch.empty(numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=arith)
    out = buf[:numel].view(N, C, Hp, sw, wq)
    out_s = out.stride()

    # _pad_split_cast_kernel's load must stay mask-free (a masked or clamped
    # index folded through reshape + tl.split silently zeroes the tile on this
    # backend), so rows must divide BLOCK_ROWS and the halo width must cover a
    # whole BLOCK_Q tile.  ROWS that do not divide fall back to _pad_split_width.
    if rows % block_rows != 0:
        return (
            _pad_split_width(input.to(arith), padding, sw, dw, kw, out_w, block_w),
            None,
        )
    wq_pad = triton.cdiv(wq, block_q) * block_q
    halo = _pad_input(input, padding, wq_pad * sw - Wp).contiguous()
    grid = (rows // block_rows, triton.cdiv(wq, block_q))
    _pad_split_cast_kernel[grid](
        halo,
        out,
        rows,
        wq,
        wq_pad * sw,  # halo_row_stride
        out_s[2],  # out_row_stride == sw * wq
        out_s[3],  # out_r_stride == wq
        SW=sw,
        BLOCK_ROWS=block_rows,
        BLOCK_Q=block_q,
        CAST_TO_FP32=(arith == torch.float32 and input.dtype != torch.float32),
        num_warps=_NUM_WARPS,
    )
    return out, None


def _pick_blocks(ow, oc_per_group, block_w_cap=None):
    """Pick (BLOCK_OC, BLOCK_W) to minimise the bytes the kernel loads per MAC.

    One program loads a (BLOCK_C, BLOCK_W) input tile and a (BLOCK_OC, BLOCK_C)
    weight tile per tap and reuses them for BLOCK_OC * BLOCK_W outputs, so the
    load is BLOCK_C * (BLOCK_W + BLOCK_OC) elements for BLOCK_C * BLOCK_W *
    BLOCK_OC MACs -- that is, ``1/BLOCK_W + 1/BLOCK_OC`` bytes per MAC, times
    the element size.  The channel tile BLOCK_C cancels.

    ``_BLOCK_ELEMS`` bounds BLOCK_OC * BLOCK_W, so the two are alternatives and
    the split between them matters: they are symmetric in the formula but not in
    what pins them, because BLOCK_W is the output row and BLOCK_OC is capped by
    the output channels.  This takes the whole row and gives the channel tile
    only the remainder.

    That direction is the opposite of what this did, and it was a walk down the
    powers of two from the full row, taking a step whenever ``1/BLOCK_W +
    1/BLOCK_OC`` improved.  On the model above that is the same product either
    way, but the model is incomplete: it counts the bytes in a load and not the
    shape of one, and BLOCK_W is the run length.  Measured on the device, whole
    call, with the budget held at its own value so the tiles differ only in the
    split -- (32,64,512)/k3, bf16:

      BLOCK_W    128    256    512
      BLOCK_OC    64     64     64
      BLOCK_C     64     32     16      (what _pick_block_c allows)
      us        18.1   16.0   12.7      (1.43x)

    so the widest row is the fastest of the three even though each has the same
    BLOCK_C * BLOCK_W.  The walk reached (128, 64) here -- it moved one step off
    the full row because that is where the objective above first improved, and
    then stopped one step later.
    """
    cap_oc = min(_BLOCK_OC_MAX, max(1, triton.next_power_of_2(oc_per_group)))
    block_w = min(block_w_cap or _BLOCK_W_MAX, max(1, triton.next_power_of_2(ow)))
    block_oc = min(cap_oc, max(1, _BLOCK_ELEMS // block_w))
    return block_oc, block_w


def _can_use_dot(block_oc, block_w, c_in):
    """Whether the whole tile satisfies tl.dot's minimum dimension.

    A shape that fails this falls back to the FMA kernel rather than padding up
    to 16, and that was re-measured rather than assumed, because the kernel
    *can* pad: _pick_block_c could floor BLOCK_C at 16 and NEED_CMASK would mask
    the surplus channels to zero.  Written that way, (8,3,224,224)/k3 -- the
    16-output-channel, 3-input-channel case, where only the contraction axis is
    short -- goes from 596 us to 974 us in bf16 and 610 to 973 in fp16, against
    an FMA form at 596/610.  The cube is not far enough ahead of the vector unit
    here to pay for 16/3 the arithmetic, and it is 1.6x behind once it is.

    Only fp32 comes out ahead (502 us to 451), where the FMA chain is at its
    slowest.  The 2D launcher now exploits exactly that one case: it forces
    ``use_dot`` when the output tile is dot-sized and pins ``arith`` to fp32 --
    see the note in _direct_conv2d.

    The channel count is asked for *rounded up*, because that is the tile the
    kernel actually builds: _pick_block_c takes the next power of two, so a
    c_in of 12 already yields BLOCK_C of 16 and a real dot.  Testing raw c_in
    kept (16,24,2048)/k7/s1/g2 -- 12 channels, one power of two short -- on the
    FMA arm, where 49 taps of 12 channels are 588 rank-one updates on the
    vector unit against one padded dot per tap: 617 us to 144 us, 4.3x.

    Rounding up is only allowed to *double* the tile, i.e. c_in of 8 and above.
    A c_in of 3 would round to 4 and then have to be padded the rest of the way
    to 16, which is the 16/3 arithmetic the numbers above are about; that one
    keeps its FMA arm here, but _direct_conv2d overrides this return value for
    the fp32 case above.
    """
    tile_c = triton.next_power_of_2(c_in)
    if c_in >= _DOT_MIN // 2:
        tile_c = max(tile_c, _DOT_MIN)
    return min(block_oc, block_w, tile_c) >= _DOT_MIN


def _pick_block_c(c_in, use_dot, block_w):
    """Channel tile for the reduction, in elements.

    Only meaningful on the dot path; the FMA path walks one channel at a time.
    Capped at _BLOCK_C_MAX rather than left at c_in so a very wide channel dim
    is reduced in a few dots instead of one impractically large tile, and
    rounded up so a channel count between powers of two still gets one dot with
    a masked tail rather than a mostly-empty second block.

    Then shrunk, if it has to be, until the input tile fits the unified-buffer
    budget -- see _UB_TILE_MAX.  The channel axis is the one to give up rather
    than BLOCK_W: on (64,48,1024)/k5/s2 halving the channel tile costs 0.545 ms
    against 0.593 ms for halving the width, and on (64,64,1024)/k3/s2 the two
    tie at 0.362 ms.
    """
    if not use_dot:
        return 1
    # Floored at _DOT_MIN, matching _can_use_dot: a c_in of 8 reaches this path
    # only because padding it to 16 was judged worth the arithmetic, and
    # BLOCK_C of 8 would not be a dot at all.
    block_c = max(_DOT_MIN, min(_BLOCK_C_MAX, triton.next_power_of_2(c_in)))
    while block_c > _DOT_MIN and block_c * block_w > _UB_TILE_MAX:
        block_c //= 2
    return block_c


@libentry()
@triton.jit
def _densify_kernel(
    w_ptr,
    out_ptr,
    N,
    ND,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter the (OC, 1, *k) weight onto the main diagonal of (OC, C, *k).

    One flat walk over the destination, so there is no zeros pass and no
    arange/IndexPutV2 pair: the element is either on the diagonal, and loaded
    from ``w[oc, 0, t]``, or it is a zero that this store writes directly.

    The source address is in bounds for every lane, masked or not -- ``oc`` and
    ``t`` both come out of a division of an index that is itself inside the
    destination, so ``oc * T + t <= OC * T`` -- which is why this needs no
    slack on the input.
    """
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = n < N
    t = n % T
    c = (n // T) % C
    oc = n // (T * C)
    keep = ok & (c == oc) & (oc < ND)
    v = tl.load(w_ptr + oc * T + t, mask=keep, other=0.0)
    tl.store(out_ptr + n, v, mask=ok)


@libentry()
@triton.jit
def _prep_weight_kernel(
    w_ptr,
    out_ptr,
    S_OC: tl.constexpr,
    C: tl.constexpr,
    OC: tl.constexpr,
    T: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CT: tl.constexpr,
):
    """Copy (OC, C, *k) into the (T, C, OC) the conv kernels read.

    The tap and channel axes are flattened into one ``ct = c * T + t`` so the
    tile can be a plain 2-D block whose fast axis is the *source's* contiguous
    one -- ``ct`` has stride 1 in (OC, C, *k), which is what lets the MTE issue
    one run per row instead of a gather.  That block is then put through
    ``tl.trans`` and stored into the destination, whose fast axis is ``oc``.
    The store side scatters one row per (c, t), which is the unavoidable half
    of a transpose; the load side does not have to scatter as well, and that is
    the whole reason this beats the vendor copy it replaces (see _prep_weight).

    ``S_OC`` is the source's channel-quad stride; ``C``, ``OC`` and ``T`` are
    constexpr rather than launch scalars so that the ``ct // T`` below is a
    compile-time multiply rather than the per-lane emulated division the backend
    falls back to for a runtime divisor.
    """
    ct = tl.program_id(0) * BLOCK_CT + tl.arange(0, BLOCK_CT)
    oc = tl.program_id(1) * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ct_ok = ct < C * T
    oc_ok = oc < OC
    c = ct // T
    t = ct - c * T
    # Addressed with ``ct`` and not with ``c * T + t``.  The two are the same
    # number and are not the same address expression: ``c`` and ``t`` come out of
    # a division, so the sum has no provable stride and the backend issues it as
    # a gather -- 69.4 us against 10.0 us on (32,64,512)/k3, 70.2 us against
    # 11.1 us on (8,256,64,64)/k3, with a flat copy of the same volume at 1.6
    # and 2.6 us for scale.  Written as ``ct`` the row is one contiguous run and
    # the MTE issues it as one.
    v = tl.load(
        w_ptr + oc[:, None] * S_OC + ct[None, :],
        mask=oc_ok[:, None] & ct_ok[None, :],
        other=0.0,
    )
    if OUT_FP32:
        v = v.to(tl.float32)
    tl.store(
        out_ptr + t[:, None] * (C * OC) + c[:, None] * OC + oc[None, :],
        tl.trans(v),
        mask=ct_ok[:, None] & oc_ok[None, :],
    )


def _densify_depthwise(weight, cin):
    """(OC, 1, *k) -> (OC, OC, *k) with the tap weights on the main diagonal.

    Depthwise is a dense convolution whose weight happens to be block diagonal
    with 1x1 blocks, so writing those blocks out and running the ordinary dense
    kernel computes exactly the same sum -- ``sum_c W[c, oc] * x[c]`` over a W
    that is zero off the diagonal is ``w[oc] * x[oc]``.  Every product is exact
    and no term is dropped; only the order the accumulator adds them in changes.

    It costs ``groups`` times the arithmetic, which sounds like the wrong trade
    until you price the alternative.  ``groups == C_in`` collapses BLOCK_OC to 1,
    so the FMA kernel runs one output channel per program with a scalar weight
    per tap: on (16,32,56,56)/k3/g32 that is 4904 us for 462 MMAC, against
    2.36 TMAC/s for the same arithmetic on the cube -- and the same shape's
    dense neighbour (16,32,32,32)/k3 does its 151 MMAC in 64 us.  Scaling that
    by output count puts the dense form at ~196 us, 25x under the FMA one.  The
    block-diagonal weight is 9216 elements here, which is why the build is two
    small tensor ops and not something worth fusing.

    Only the 1x1-block case is worth this: a group with more than one input or
    output channel already clears tl.dot's minimum dimension on its own and
    takes the dot path without any padding of the arithmetic.

    ``torch.zeros`` followed by ``out[idx, idx] = weight[:, 0]`` is the obvious
    build and was the one this used: it is three vendor kernels -- a ZerosLike,
    a Range for the index, and an IndexPutV2 -- and on (16,32,1024)/k3/g32 they
    come to 16.9 us against a whole call of 63.9 us.  _densify_kernel does the
    same fill in one.
    """
    OC, _, *k = weight.shape
    t = 1
    for size in k:
        t *= size
    numel = OC * cin * t
    buf = torch.empty(numel + _SLACK_ELEMS, device=weight.device, dtype=weight.dtype)
    out = buf[:numel].view(OC, cin, *k)
    _densify_kernel[(triton.cdiv(numel, _PREP_BLOCK),)](
        weight.contiguous(),
        out,
        numel,
        min(OC, cin),
        C=cin,
        T=t,
        BLOCK=_PREP_BLOCK,
        num_warps=_NUM_WARPS,
    )
    return out


# First-axis extent _prep_blocks picks below the tile-picking crossover, and
# the same extent above it.  Both are sweep results; see _prep_blocks.
_PREP_CT_SMALL = 8
_PREP_CT_LARGE = 16
_PREP_CT_MAX_ELTS = 256


def _prep_blocks(ct_n, oc):
    """Tile for _prep_weight_kernel, from a sweep over every weight in the suite.

    The rule this replaces -- the next power of two, capped at _PREP_BLOCK --
    takes the *largest* tile each axis allows, and on these weights that is one
    64x64 program holding 12288 of them: 9.7 us on the 64x64x3 weight of
    (32,64,512), against 3.8 us for a 16x64 tile over the same tensor, and
    (256,256,3), the largest weight here, is 19.8 us at 64x64 against 14.8 us at
    16x64.  Neither tensor is bandwidth bound -- (256,256,3) is 512 KB against
    the 3 MB its rows and columns span -- so the tile that minimises programs is
    not the tile that minimises time.  What does is a first-axis extent of 8 or
    16, whichever side of 256 elements on that axis, with the second axis left
    as wide as it was: across the suite's shapes that pairing is at or within
    1 us of the best of the ~30 (ct, oc) pairs tried per shape, and the sweep's
    optimum moves with the axis length while this does not.

    The second axis is *not* free to shrink, which is why it keeps the old rule.
    It is the kernel's store row -- contiguous, ``oc`` elements wide, one per
    row of the tile -- so its width is the run length the MTE writes, and 8 is
    where that collapses: the same sweep puts a 64-element store row at 3.3 us
    and an 8-element one at 16 us on (32,64,512), and no first-axis extent
    recovers it.  Wide is also bounded: 64 elements is 128 B, one cache line.
    """
    block_oc = min(_PREP_BLOCK, triton.next_power_of_2(oc))
    if ct_n <= _PREP_CT_MAX_ELTS:
        block_ct = _PREP_CT_SMALL
    else:
        block_ct = _PREP_CT_LARGE
    return min(block_ct, triton.next_power_of_2(ct_n)), block_oc


def _prep_weight(weight, out_dtype=None):
    """Move the output-channel axis last and make it contiguous.

    This is the single largest win in the file, and it is a layout fix rather
    than an arithmetic one.  The kernels below read one weight vector per
    (tap, channel) pair, indexed by output channel.  In the native (OC, C, KH,
    KW) layout that vector has stride C*KH*KW -- a gather, in the innermost loop,
    once per tap and channel.  Permuting to (KH, KW, C, OC) makes it unit-stride.

    Measured on (32,64,128,128)/3x3/pad2, identical kernel otherwise:
    140.8 ms native layout vs 36.1 ms permuted, 3.9x.  The cost is one small
    transpose per call on a tensor that is at most a few hundred thousand
    elements, against one gather per program per tap per channel.

    The permutation lands in a buffer with _SLACK_ELEMS to spare because the
    kernel's masked output-channel lanes still compute an address past the end
    of it.  The tail beyond the tensor is never read or written -- the mask
    suppresses both -- but the MTE faults on it before the mask is consulted if
    it lands outside the allocation; see _slack.

    ``weight.permute(...)`` then ``out.copy_(permuted)`` is the obvious build
    and was the one this used.  aclnn lowers that copy to an InplaceCopy that
    carries a Transpose kernel, and the Transpose is *flat* in size where it is
    not outright slow: 27.0 us on the 18432-element weight of
    (32,64,128,128)/k3 against a triton kernel floor of 1.5 us, and 10.4 us on
    the 16 KB, 256 KB and 1 MB tensors measured off the suite.  It is the
    second largest single kernel on the 2-D core case after the convolution
    itself.  _prep_weight_kernel does the same transpose in one launch, and it
    takes ``out_dtype`` so the ``weight.to(arith)`` cast the FMA arm needs
    rides along with it instead of costing a second kernel -- see
    _arith_dtype for when that cast happens at all.
    """
    oc, c = weight.shape[0], weight.shape[1]
    t = 1
    for size in weight.shape[2:]:
        t *= size
    out_dtype = weight.dtype if out_dtype is None else out_dtype
    numel = oc * c * t
    buf = torch.empty(numel + _SLACK_ELEMS, device=weight.device, dtype=out_dtype)
    # Kept as (*k, C, OC) rather than flattened to (T, C, OC): the two are the
    # same buffer in the same order, and the launchers read the channel stride
    # back off with ``wt.stride(weight.dim() - 2)``, which needs the tap axes
    # still spelled out.
    out = buf[:numel].view(*weight.shape[2:], c, oc)
    block_ct, block_oc = _prep_blocks(c * t, oc)
    _prep_weight_kernel[(triton.cdiv(c * t, block_ct), triton.cdiv(oc, block_oc))](
        weight.contiguous(),
        out,
        S_OC=c * t,
        C=c,
        OC=oc,
        T=t,
        OUT_FP32=out_dtype == torch.float32,
        BLOCK_OC=block_oc,
        BLOCK_CT=block_ct,
        num_warps=_NUM_WARPS,
    )
    return out


def _slack(out_c_stride, block_oc, block_w):
    """Spare elements to append to an allocation the kernels index directly.

    A lane whose index is past the end of its tile keeps its address: the masks
    suppress the *access*, not the address computation, and the device's MTE
    faults on an address outside the allocation with 507015 "The DDR address of
    the MTE instruction is out of range" before the mask is consulted.  The
    overshoot is bounded by the tile -- BLOCK_W along the innermost axis, and a
    whole output plane per masked output-channel block on the channel axis --
    and rounding up to _SLACK_ELEMS covers the allocator's own alignment, which
    is what makes the fault look intermittent: whether a call survives depends
    on where the allocator happened to put the block.
    """
    return max(_SLACK_ELEMS, (block_oc - 1) * out_c_stride + block_w)


@libentry()
@triton.jit
def _direct_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    H,
    W,
    OC,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_h_stride,
    in_r_stride,
    in_w_stride,
    w_c_stride,
    out_n_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NEED_CMASK: tl.constexpr,
    USE_DOT: tl.constexpr,
    W_SPLIT: tl.constexpr,
    RUNTIME_TAPS: tl.constexpr,
    W_OC_STRIDE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    # One (batch, output row) pair per row id; long rows are cut into segments.
    num_seg = tl.cdiv(OW, BLOCK_W)
    n = pid_row // (OH * num_seg)
    rem = pid_row % (OH * num_seg)
    oh = rem // num_seg
    seg = rem % num_seg

    ow = seg * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok = ow < OW

    in_row = input_ptr + n * in_n_stride
    in_group = pid_group * C_IN * in_c_stride

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)
    # Tap-major loop order: the tap offset and the tap pointer depend only on
    # (kh, kw), so hoisting them out of the channel loop keeps one address
    # computation per tap instead of one per tap and channel.
    #
    # The tap indices here are the *unpadded* ones -- ``oh * SH + kh * DH``, not
    # ``- PH`` -- so the launcher's zero halo is what makes them valid: the
    # padded tensor's element [ih + PH, iw + PW] is the input's [ih, iw], and
    # every tap that falls in the halo reads a real zero.  That is why the loads
    # below carry no mask and no bounds test.  It is also why they are safe: a
    # lane whose ow is past OW reads whatever the halo or the following row
    # holds, and its column of the accumulator is thrown away by the store's
    # w_ok.
    #
    # This is not a stylistic choice.  The same address expressed as
    # ``clamp(ow * SW + kw * DW - PW, 0, W - 1)`` -- the obvious way to keep a
    # masked lane inside the tensor -- takes the address out of the affine form
    # the backend's axis analysis needs, and the kernel then does the same work
    # an order of magnitude slower: measured 616 ms against 3.8 ms on
    # (32,64,128,128)/k3/s2, at 97% cube utilisation either way.  Materialising
    # the halo costs one pass over the input instead.
    #
    # Both arms of this kernel live in one function, and the split into two
    # was tried and reverted.  A constexpr `if` prunes its dead arm -- checked
    # directly, not assumed -- and neither arm pays for the other's presence:
    # built as two kernels against one, alternating inside a single process so
    # the drift between runs cancels, (8,3,224,224)/16/k3 and
    # (32,64,128,128)/32/k3 both come out 1.000x in either form, on both arms.
    # A kernel per arm would duplicate this nest into two copies that then
    # drift; the 3D launcher losing its depthwise lift while the 2D one kept
    # its own is what that costs, and that was 7x on the depthwise shape.
    if RUNTIME_TAPS:
        # Large kernels walk their taps in a *runtime* loop: KH*KW unrolled
        # bf16 dots hang the device (see _arith_dtype), so any tap count past
        # _DOT_TAPS_MAX must go through a loop, and a runtime loop has one dot in
        # the unrolled body however many taps it makes.  The loop is used for
        # every large-kernel dot case -- in fp32 on the split-width path, in bf16
        # elsewhere, chosen by the launcher via _arith_dtype -- and the fp32 form
        # is itself a win over the unrolled form because it staggers the tap
        # loads instead of bursting them all at once.  The address arithmetic is
        # runtime too (kh and kw come out of a division), a handful of scalar ops
        # against what the unrolled form would cost.  RUNTIME_TAPS is only ever
        # set with USE_DOT, so the FMA arm is absent here.
        for t in range(KH * KW):
            kh = t // KW
            kw = t - kh * KW
            ih = oh * SH + kh * DH
            if W_SPLIT:
                tap_in = (
                    in_group
                    + in_row
                    + ih * in_h_stride
                    + ((kw * DW) % SW) * in_r_stride
                    + ow
                    + (kw * DW) // SW
                )
            else:
                iw = ow * SW + kw * DW
                tap_in = in_group + in_row + ih * in_h_stride + iw * in_w_stride
            tap_w = weight_ptr + t * C_IN * w_c_stride + oc_glob * W_OC_STRIDE
            cc = tl.arange(0, BLOCK_C)
            for cb in range(0, C_IN, BLOCK_C):
                if NEED_CMASK:
                    c_mask = (cb + cc) < C_IN
                    x = tl.load(
                        tap_in + (cb + cc)[:, None] * in_c_stride,
                        mask=c_mask[:, None],
                        other=0.0,
                    )
                    w = tl.load(
                        tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                        mask=oc_mask[:, None] & c_mask[None, :],
                        other=0.0,
                    )
                else:
                    x = tl.load(tap_in + (cb + cc)[:, None] * in_c_stride)
                    w = tl.load(
                        tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                        mask=oc_mask[:, None],
                        other=0.0,
                    )
                acc = tl.dot(w, x, acc)
    else:
        for kh in tl.static_range(KH):
            ih = oh * SH + kh * DH
            for kw in tl.static_range(KW):
                if W_SPLIT:
                    # The launcher reshaped the halo's width axis to (W/SW, SW) and
                    # transposed it, so tap kw lives in plane (kw*DW) % SW at column
                    # ow + (kw*DW)//SW.  Unit stride in ow, which is the whole point;
                    # see _pad_split_width.  Both plane offsets are Python ints here
                    # because kw is unrolled.
                    tap_in = (
                        in_group
                        + in_row
                        + ih * in_h_stride
                        + ((kw * DW) % SW) * in_r_stride
                        + ow
                        + (kw * DW) // SW
                    )
                else:
                    iw = ow * SW + kw * DW
                    tap_in = in_group + in_row + ih * in_h_stride + iw * in_w_stride
                # Weight is (KH, KW, C, OC), so the tap selects a plane and the
                # output channel is the unit-stride axis.
                tap_w = (
                    weight_ptr
                    + (kh * KW + kw) * C_IN * w_c_stride
                    + oc_glob * W_OC_STRIDE
                )
                if USE_DOT:
                    # One (BLOCK_OC, BLOCK_C) x (BLOCK_C, BLOCK_W) dot per tap and
                    # channel block, which is what puts this on the cube. Reducing
                    # the same work as FMA instead runs it on the vector unit:
                    # measured 3.29 ms vs 28.0 ms on the 2D core case.
                    cc = tl.arange(0, BLOCK_C)
                    for cb in range(0, C_IN, BLOCK_C):
                        if NEED_CMASK:
                            c_mask = (cb + cc) < C_IN
                            x = tl.load(
                                tap_in + (cb + cc)[:, None] * in_c_stride,
                                mask=c_mask[:, None],
                                other=0.0,
                            )
                            w = tl.load(
                                tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                                mask=oc_mask[:, None] & c_mask[None, :],
                                other=0.0,
                            )
                        else:
                            x = tl.load(tap_in + (cb + cc)[:, None] * in_c_stride)
                            w = tl.load(
                                tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                                mask=oc_mask[:, None],
                                other=0.0,
                            )
                        acc = tl.dot(w, x, acc)
                else:
                    # Deliberately `range`, not `tl.static_range`: see the note above.
                    for c in range(C_IN):
                        x = tl.load(tap_in + c * in_c_stride)
                        w = tl.load(tap_w + c * w_c_stride, mask=oc_mask, other=0.0)
                        acc += x[None, :] * w[:, None]

    tl.store(
        output_ptr
        + oc_glob[:, None] * out_c_stride
        + n * out_n_stride
        + oh * out_h_stride
        + ow[None, :] * out_w_stride,
        acc,
        mask=oc_mask[:, None] & w_ok[None, :],
    )


@libentry()
@triton.jit
def _flat_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    in_n_stride,
    in_c_stride,
    out_n_stride,
    out_plane,
    tot_p,
    wp,
    C_IN,
    OC,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NEED_OCMASK: tl.constexpr,
    NEED_CMASK: tl.constexpr,
):
    """Stride-1 conv2d over a *flat* (oh, ow) index, so x runs long and contiguous.

    Probe of what this fixes.  The tap address of output position (oh, ow) is

        (oh + kh) * Wp + (ow + kw)   =   p + kh * Wp + kw,   p = oh * Wp + ow

    a *constant* shift -- provided the padded input's row stride and the output
    plane's row stride are both Wp.  So a program can own a flat interval of p
    spanning many rows, and the x load becomes (BLOCK_C, BLOCK_M) with BLOCK_M*2
    contiguous bytes per row instead of BLOCK_W*2.

    That run length is the whole point.  Measured on this device, isolating one
    load pattern, GB/s against a 1753 GB/s fully-flat copy:

        run     256 B   512 B   1024 B   2048 B
        GB/s      382     623      859     1367

    It is the length of the innermost contiguous run, and at W=128/k3/p2 the
    blocked kernel's run is 2*OW bytes.  Flat takes it to BLOCK_M.

    The tap count is *not* free, though an isolated measurement says it is.
    Reading one tile nine times at one-element offsets costs 0.98x of reading it
    once, but scaling KH*KW over 1/9/25/49 on a fixed shape *inside this kernel*
    costs 1.00x / 8.75x / 19.83x / 37.61x, with the effective bandwidth pinned at
    200-240 GB/s instead of rising.  Pinned bandwidth as the tap count grows is
    what an MTE2-issue bound looks like: the taps are paid for in request slots,
    not absorbed by cache.  The isolated probe was missing the GM pressure of
    the real call, in the same way as the ones recorded on _direct_conv2d_kernel
    -- treat any tile or tap measurement taken outside the whole call as a
    hypothesis, not a result.

    On (32,64,128,128)/oc32/k3/p2 the conv goes 1709 us -> 533 us, 11.7 -> 37.4
    TFLOP/s.  The ceiling is now the cube, not the MTE: 19.9 GFLOP over 533 us.

    The price is the output layout.  p is flat over a plane that is Wp wide, so
    the result has to be written Wp-strided and compacted back to OW columns
    afterwards -- see _compact_plane_kernel.  Nothing else changes: no mask on
    the tap load (the halo supplies the zeros, exactly as in the blocked
    kernel), and the address stays affine in p, which is what the backend's axis
    analysis needs.

    Wp must be the *unwidened* padded width, W + 2*PW.  The blocked kernel
    widens every padded row so the last tile's masked lanes land inside it; here
    the only overshoot is the last program's, so the room is a single tail on
    the whole allocation instead of 126 columns on every row.  That also halves
    the halo: Wp goes 258 -> 132 on (32,64,128,128), so the strided interior
    copy that builds it runs at 97% row density instead of 50%.
    """
    pid_p = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ng = tl.program_id(2)
    if GROUPS == 1:
        n = pid_ng
        pid_group = 0
    else:
        n = pid_ng // GROUPS
        pid_group = pid_ng % GROUPS

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    p = pid_p * BLOCK_M + tl.arange(0, BLOCK_M)
    cc = tl.arange(0, BLOCK_C)

    base = input_ptr + n * in_n_stride + pid_group * C_IN * in_c_stride
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            off = p + (kh * DH) * wp + kw * DW
            tap_w = weight_ptr + (kh * KW + kw) * C_IN * OC + oc_glob
            for cb in range(0, C_IN, BLOCK_C):
                x = tl.load(base + (cb + cc)[:, None] * in_c_stride + off[None, :])
                # The channel mask sits on the *weight*, not on x, and only its
                # last block can fail it: a lane of x that overshoots C_IN reads
                # the next channel, which is garbage but in-bounds given the
                # tail, and the zeroed weight discards it.  Masking x instead
                # would put a compare on the load that matters (BLOCK_C x
                # BLOCK_M) to save a compare on the one that does not.
                w_off = tap_w[:, None] + (cb + cc)[None, :] * OC
                if NEED_CMASK or NEED_OCMASK:
                    w = tl.load(
                        w_off,
                        mask=oc_mask[:, None] & ((cb + cc)[None, :] < C_IN),
                        other=0.0,
                    )
                else:
                    w = tl.load(w_off)
                acc = tl.dot(w, x, acc)

    # ``oc_mask`` is in the store mask for the reason the channel overshoot is:
    # a lane past OC keeps its address, and the output plane's channel stride
    # walks it straight into the next image.  Nothing upsets this in 2-D because
    # _pick_flat_blocks derives BLOCK_OC from the channel count and the two only
    # disagree above _BLOCK_OC_MAX -- but the 3-D launcher raises BLOCK_OC to
    # _DOT_MIN on shapes _DOT_MIN/2 wide, and there the surplus lanes measured a
    # 0.68-0.93 relative error scattered over the *following* image.
    tl.store(
        output_ptr + n * out_n_stride + oc_glob[:, None] * out_plane + p[None, :],
        acc,
        mask=(p < tot_p)[None, :] & oc_mask[:, None],
    )


@libentry()
@triton.jit
def _compact_plane_kernel(
    src_ptr,
    dst_ptr,
    OW,
    ROWS,
    WP: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """(rows, Wp) -> (rows, OW), dropping the Wp - OW slack columns of each row.

    Small and unavoidable, and it costs far more than the bytes it moves: 156 us
    for 69 MB on (32,64,128,128), 443 GB/s against 1753 GB/s for a flat copy.

    That is the row stride, not the mask.  Swept at fixed volume, every form
    lands in the same band -- OW=132 unmasked and contiguous 52 us, OW=130
    masked 51-64 us, OW=128 exactly-masked 54-64 us, OW=126 masked 51-55 us --
    so a store that needs no mask at all is no faster than one that does.  What
    does cost is giving each row its own address: one row per program with a 1-D
    contiguous load and a 1-D contiguous store is 716 us, five times *worse*,
    and the same shape with a per-lane // and % to rebuild the column index is
    55.9 ms.  A strided 2-D tile is the best form available; this is it.

    So do not try to fuse the compaction into the kernel that produces the
    plane.  Writing ``oh = p // Wp; ow = p - oh * Wp`` into _flat_conv2d_kernel's
    store is bit-identical to the two-pass form and 633-742x slower on the four
    flat benchmark cases -- 384448 us against 599 us on (8,256,64,64), 773631
    against 1043 on (32,64,128,128).  A constexpr divisor does not save it, so
    the cost is not the division: ``oh * OW + ow`` is not affine in ``p``, the
    backend can no longer prove the store's stride, and it emits a gather.  Same
    mechanism as the ``//`` in _prep_weight, one pipe over.
    """
    r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    m = (c < OW)[None, :] & (r < ROWS)[:, None]
    v = tl.load(src_ptr + r[:, None] * WP + c[None, :], mask=m, other=0.0)
    tl.store(dst_ptr + r[:, None] * OW + c[None, :], v, mask=m)


@libentry()
@triton.jit
def _direct_conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    H,
    W,
    OC,
    OD,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_d_stride,
    in_h_stride,
    in_w_stride,
    w_c_stride,
    out_n_stride,
    out_c_stride,
    out_d_stride,
    out_h_stride,
    out_w_stride,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NEED_CMASK: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    num_seg = tl.cdiv(OW, BLOCK_W)
    n = pid_row // (OD * OH * num_seg)
    rem = pid_row % (OD * OH * num_seg)
    od = rem // (OH * num_seg)
    rem = rem % (OH * num_seg)
    oh = rem // num_seg
    seg = rem % num_seg

    ow = seg * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok = ow < OW

    in_row = input_ptr + n * in_n_stride
    in_group = pid_group * C_IN * in_c_stride

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)
    # Channel-major, unlike the 2D kernel: putting the runtime channel loop
    # around 27 unrolled taps instead of 9 sends the Ascend compiler into the
    # same kind of multi-minute unroll the note above describes -- measured
    # stuck for > 15 min on (1,16,4,4,4) / 3x3x3.  Nine taps is under the
    # limit; twenty-seven is not.
    #
    # Both arms are in this one kernel.  A dot-only sibling holding just the
    # if-branch was tried and removed: a constexpr `if` prunes its dead arm, so
    # neither arm pays for the other, and every 3D shape measured the same
    # either way -- all fifteen (case, dtype) pairs within 1.5%, over three
    # runs of each form.  Two kernels that duplicate this addressing block are
    # what let the 3D launcher lose the depthwise lift the 2D one had, and that
    # was 7x on the depthwise shape; see _densify_depthwise.
    #
    # The tap indices are unpadded and the loads unmasked, for the reason given
    # in the 2D kernel: the launcher materialises the zero halo, so the halo
    # supplies the padding and every tap address is in bounds.
    if USE_DOT:
        cc = tl.arange(0, BLOCK_C)
        for t in range(KD * KH * KW):
            kd = t // (KH * KW)
            kh = (t // KW) % KH
            kw = t % KW
            idd = od * SD + kd * DD
            ih = oh * SH + kh * DH
            iw = ow * SW + kw * DW
            tap_in = (
                in_group
                + in_row
                + idd * in_d_stride
                + ih * in_h_stride
                + iw * in_w_stride
            )
            tap_w = weight_ptr + t * C_IN * w_c_stride + oc_glob
            for cb in range(0, C_IN, BLOCK_C):
                if NEED_CMASK:
                    c_mask = (cb + cc) < C_IN
                    x = tl.load(
                        tap_in + (cb + cc)[:, None] * in_c_stride,
                        mask=c_mask[:, None],
                        other=0.0,
                    )
                    w = tl.load(
                        tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                        mask=oc_mask[:, None] & c_mask[None, :],
                        other=0.0,
                    )
                else:
                    x = tl.load(tap_in + (cb + cc)[:, None] * in_c_stride)
                    w = tl.load(
                        tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                        mask=oc_mask[:, None],
                        other=0.0,
                    )
                acc = tl.dot(w, x, acc)
    else:
        for c in range(C_IN):
            in_c_off = in_group + in_row + c * in_c_stride
            for kd in tl.static_range(KD):
                idd = od * SD + kd * DD
                for kh in tl.static_range(KH):
                    ih = oh * SH + kh * DH
                    for kw in tl.static_range(KW):
                        iw = ow * SW + kw * DW
                        x = tl.load(
                            in_c_off
                            + idd * in_d_stride
                            + ih * in_h_stride
                            + iw * in_w_stride
                        )
                        w = tl.load(
                            weight_ptr
                            + ((kd * KH + kh) * KW + kw) * C_IN * w_c_stride
                            + oc_glob
                            + c * w_c_stride,
                            mask=oc_mask,
                            other=0.0,
                        )
                        acc += x[None, :] * w[:, None]

    tl.store(
        output_ptr
        + oc_glob[:, None] * out_c_stride
        + n * out_n_stride
        + od * out_d_stride
        + oh * out_h_stride
        + ow[None, :] * out_w_stride,
        acc,
        mask=oc_mask[:, None] & w_ok[None, :],
    )


@libentry()
@triton.jit
def _flat_conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    in_n_stride,
    in_c_stride,
    in_d_stride,
    out_n_stride,
    out_c_stride,
    out_d_stride,
    tot_p,
    RS,
    C_IN,
    OC,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    GROUPS: tl.constexpr,
    OD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NEED_OCMASK: tl.constexpr,
    NEED_CMASK: tl.constexpr,
):
    """Stride-1 conv3d over a *flat* (oh, ow) index, one depth slice per program.

    The 2-D kernel's identity, applied one dimension at a time.  A tap of output
    position (od, oh, ow) reads the padded input at

        (od + kd*DD)*Dp*Hp*Wp + (oh + kh*DH)*Wp + (ow + kw*DW)
      = od*Dp*Hp*Wp + [(oh*Wp + ow) + kd*DD*Dp*Hp*Wp + kh*DH*Wp + kw*DW]

    so with the output's flat index taken over one depth slice,
    ``p = oh*Wp + ow``, every tap is again a constant shift of ``p``.  The depth
    axis rides on the grid as ``od`` and costs nothing -- input and output spell
    it with different strides, which is fine only because it is outside ``p``.

    That is the whole fix for 3-D, and it is the same one 2-D needed.  The
    blocked 3-D kernel's innermost run is BLOCK_W fp32 elements -- 16 of them,
    64 bytes, at W=16 -- and the MTE pays for the cache lines a run touches
    rather than the bytes it keeps.  Measured through this entry point on the
    (2,16,16,16,16)/k3 shape, 512 depth slices, only the run varying:

        run       64 B   128 B   256 B   512 B   1024 B   2048 B
        us        958     605     564     327      286      199
        GB/s      310     515     603    1038     1587     2281

    7.4x end to end, and the taps stop costing: at 2048 B runs nine taps move
    2022 GB/s against 342 for one, i.e. they hit cache once the working set
    fits.  Flat's BLOCK_M is 256-1024 elements here (the (oh, ow) plane of these
    shapes is 168-624 wide), so every 3-D case lands in the top two rows.

    Wp is the un-widened padded width, as in 2-D, and the price is the same:
    the result is written Wp-strided per depth slice and compacted back to OW
    columns by _compact_plane_kernel, which sees it as a plain
    (N*OC*OD*OH, Wp) plane because the slices are contiguous inside a channel.
    """
    pid_p = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_a = tl.program_id(2)

    n = pid_a // (GROUPS * OD)
    rem = pid_a % (GROUPS * OD)
    pid_group = rem // OD
    od = rem % OD

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    p = pid_p * BLOCK_M + tl.arange(0, BLOCK_M)
    cc = tl.arange(0, BLOCK_C)

    base = (
        input_ptr + n * in_n_stride + pid_group * C_IN * in_c_stride + od * in_d_stride
    )
    acc = tl.zeros((BLOCK_OC, BLOCK_M), dtype=tl.float32)
    for kd in tl.static_range(KD):
        for kh in tl.static_range(KH):
            h_off = kd * DD * in_d_stride + kh * DH * RS
            for kw in tl.static_range(KW):
                off = p + (h_off + kw * DW)
                tap_w = weight_ptr + ((kd * KH + kh) * KW + kw) * C_IN * OC + oc_glob
                for cb in range(0, C_IN, BLOCK_C):
                    x = tl.load(base + (cb + cc)[:, None] * in_c_stride + off[None, :])
                    # The channel mask sits on the weight, not on x, for the
                    # reason given in the 2-D kernel: the overshooting x lanes
                    # read the next channel, which the zeroed weight discards,
                    # and a masked load would cost a compare on the tile that
                    # matters to save one on the tile that does not.
                    w_off = tap_w[:, None] + (cb + cc)[None, :] * OC
                    if NEED_CMASK or NEED_OCMASK:
                        w = tl.load(
                            w_off,
                            mask=oc_mask[:, None] & ((cb + cc)[None, :] < C_IN),
                            other=0.0,
                        )
                    else:
                        w = tl.load(w_off)
                    acc = tl.dot(w, x, acc)

    # ``oc_mask`` in the store mask covers the BLOCK_OC this launcher raises to
    # _DOT_MIN, which the shape rarely fills; see the 2-D kernel.
    tl.store(
        output_ptr
        + n * out_n_stride
        + od * out_d_stride
        + oc_glob[:, None] * out_c_stride
        + p[None, :],
        acc,
        mask=(p < tot_p)[None, :] & oc_mask[:, None],
    )


# ``_pad_copy_interior_3d_kernel`` opens its width axis with a single
# ``tl.arange`` and no loop over it, so a padded row wider than this cannot use
# it and falls back to the strided copy.
_PAD_FLAT_W_MAX = 128


def _pad_input_flat(input, padding, tail):
    """``_pad_input``'s halo with the spare room as one tail on the allocation.

    ``_pad_input`` widens *every* padded row by ``tail`` so the last output
    tile's masked lanes land inside the tensor.  The flat kernel has no per-tile
    masked lanes -- its programs cover the plane contiguously and only the very
    last one overshoots -- so the rows can stay exactly ``W + 2*PW`` wide and
    the room can be a single tail past the end.

    That is worth more than the tail it saves.  On (32,64,128,128)/p2 the
    blocked path's block_w of 256 forces tail=126, so the interior copy is a
    256 B row written into a 516 B stride: ~420 GB/s.  At Wp=132 the same copy
    is a 256 B row into a 264 B stride, 97% dense, and the memset that precedes
    it is half the bytes.  Measured, that pair is 394 us against 198.

    The rows have to be un-widened for the flat identity to hold at all: the
    pad's row stride *is* the output plane's row stride, and the kernel's tap
    offset ``(oh+kh)*Wp + ow+kw`` only collapses to ``p + kh*Wp + kw`` when both
    use the same Wp.

    The 5-D interior is filled by the same affine triton pair ``_pad_input``
    uses, not by the strided ``copy_``: aclnn's gather for a 5-D destination
    lands on its **AiCpu** ViewCopy, which is a host-side element loop and costs
    200 us on a 0.5 MB interior -- against 18.7 us for the kernel pair, and
    against a whole (2,16,16,16,16) call of 50 us.  On the four flat 3-D shapes
    it was 70-92% of the call (199.8 / 669.7 / 146.8 / 136.6 us).  The 4-D
    destination stays on ``copy_``: there aclnn picks its AiCore ViewCopy, which
    at 549 GB/s beats the triton copy's 512 B rows (623 GB/s) once the
    ``torch.zeros`` memset it shares the pass with is counted.
    """
    tail = max(0, tail)
    if tail <= 0 and not any(padding):
        return input
    N, C = input.shape[0], input.shape[1]
    spatial = input.shape[2:]
    padded = [s + 2 * p for s, p in zip(spatial, padding)]
    numel = N * C
    for size in padded:
        numel *= size
    if len(spatial) == 3 and spatial[-1] <= _PAD_FLAT_W_MAX:
        D, H, W = spatial
        PD, PH, PW = padding
        Dp, Hp, Wp = padded
        out = torch.empty(numel + tail, device=input.device, dtype=input.dtype)
        _memset_kernel[(triton.cdiv(out.numel(), 1024),)](
            out, out.numel(), BLOCK=1024, num_warps=_NUM_WARPS
        )
        BLOCK_H = min(32, triton.next_power_of_2(H))
        BLOCK_W = triton.next_power_of_2(W)
        _pad_copy_interior_3d_kernel[(N * C, D, triton.cdiv(H, BLOCK_H))](
            input,
            out,
            D,
            H,
            W,
            PD,
            PH,
            PW,
            D * H * W,
            H * W,
            W,
            Dp * Hp * Wp,
            Hp * Wp,
            Wp,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            num_warps=_NUM_WARPS,
        )
        return out[:numel].view(N, C, *padded)
    if not any(padding):
        # No halo to write, only room to leave: one contiguous pass.  The buffer
        # stays uninitialised because the copy below fills *every* element of
        # ``out[:numel]`` -- with no padding there is nothing for a zero to
        # stand in for.  ``torch.zeros`` here is a whole extra write of the
        # tensor (aclnnInplaceZero 4.5 us on (16,64,56,56)/k1, against a 38 us
        # call) and the only thing it zeroes is the stretch that the copy is
        # about to overwrite anyway.  What the allocation is really buying is
        # ``tail``, the room past the end that the last program's masked-lane
        # loads point into; those lanes never store, so what they read is
        # deliberately garbage either way.
        out = torch.empty(numel + tail, device=input.device, dtype=input.dtype)
        out[:numel].copy_(input.reshape(-1))
        return out[:numel].view(N, C, *padded)
    out = torch.zeros(numel + tail, device=input.device, dtype=input.dtype)
    interior = out[:numel].view(N, C, *padded)
    window = (slice(None), slice(None)) + tuple(
        slice(p, p + s) for p, s in zip(padding, spatial)
    )
    interior[window].copy_(input)
    return interior


_FLAT_BLOCK_M = 1024
# The flat kernel's accumulator is (BLOCK_OC, BLOCK_M) in fp32 and it is what
# the unified buffer is spent on: swept on (8,256,64,64)/oc256, every tile with
# BLOCK_OC*BLOCK_M == 32768 compiles and every one above it fails, whatever
# BLOCK_C is -- (BLOCK_M=1024, BLOCK_C=16, BLOCK_OC=64) fails on the product
# alone while (256, 64, 128) and (1024, 64, 32) both run.  _BLOCK_ELEMS is the
# same 32768 the blocked launcher already tiles by, reached from the other side.
_FLAT_ACC_MAX = 32768


def _pick_flat_blocks(oc_per_group, tot_p, floor_oc=1):
    """Tile for the flat kernels: output channels first, run length second.

    BLOCK_OC sets how many output channels share one x tile, so it divides the
    x traffic; BLOCK_M only sets the length of the run each of those loads is.
    Every measurement below has both arms at the accumulator bound, so the split
    between them is the whole question, and channels win:

        (8,256,64,64)/oc256    BLOCK_OC  128  256  64  32   ->  474  403  570  845 us
        (32,64,128,128)/oc32   BLOCK_M  1024 512 256      ->  533  778  840 us

    The first shape can trade BLOCK_M away for channels and comes out 2.1x
    ahead; the second has only 32 channels to give, so BLOCK_M is all it has and
    takes the lot.  Both fall out of the same rule: spend on channels, then give
    the remainder to BLOCK_M.

    ``floor_oc`` raises the channel tile past what the shape needs, which only
    the 3-D launcher asks for: an output channel count under _DOT_MIN would
    otherwise pin BLOCK_OC below ``tl.dot``'s minimum and cost the kernel its
    dot arm.  The surplus lanes are masked to zero on the weight, so it is
    BLOCK_OC/OC times the cube arithmetic against the vector unit's 27 rank-one
    updates per tap.
    """
    block_oc = min(
        _BLOCK_OC_MAX, max(floor_oc, triton.next_power_of_2(max(1, oc_per_group)))
    )
    block_m = min(_FLAT_BLOCK_M, max(_DOT_MIN, _FLAT_ACC_MAX // block_oc))
    return block_oc, max(_DOT_MIN, min(block_m, triton.next_power_of_2(tot_p)))


# How much longer the flat kernel's innermost run has to be before the
# compaction below is worth paying for.  A gain of 1 is the trap: on
# (32,64,512) the blocked run is already OW=512 elements, flat's BLOCK_M there is
# 512 too (BLOCK_OC=64 takes the rest of the accumulator), so it buys nothing and
# still has to compact 1.07 GB of output -- 2.15 GB of traffic at ~450 GB/s.
_FLAT_RUN_GAIN = 1.5
# Ceiling on the compaction's traffic, as a multiple of the x traffic the conv
# reads to produce the output being compacted.  It is a read *and* a write of a
# (N,OC,OH,OW) plane against KH*KW passes of a (N,C,Hp,Wp) one, so the ratio is
# 2*OC*OH*OW / (KH*KW*C*Hp*Wp); on every shape that wins it lands at 0.11-0.21,
# and 0.5 leaves headroom without being reachable by the ones that lose.
#
# Do not widen it on the strength of a single case.  It was widened to 8.0 once,
# to admit (8,3,224,224)/k3 -- whose ratio is 4.66, because the compaction's
# stride Wp=226 against OW=224 makes it nearly dense while the conv's x read is
# a C=3 strided gather at 114 GB/s -- and that case did improve 1.65x (431.6 ->
# 262.2 us), but the crude byte ratio was the wrong test in more than that one
# direction: (16,32,1024) k3 also sits between the two values, and flat is much
# worse on it (46.3 -> 64.1 us; 1-D is a shape the flat identity costs a
# _compact_plane_kernel *and* a densify for, against a blocked path whose run is
# already the whole row).  Net on the suite's arithmetic mean the widening was
# -0.008: one case gained 0.066 and one lost 0.24.
_FLAT_COMPACT_MAX = 0.5


def _use_flat_2d(input, weight, padding, groups, oh, ow, use_dot):
    """Is the flat kernel both legal and worth its compaction on this shape?

    Legality first.  Stride 1, because the constant tap offset is the flat image
    of ``oh*SH + kh*DH == (oh+kh)*DH`` and only SH=1 gives it.  The dot, because
    that is the only arm the kernel has.  And the native dtype, because that is
    the regime every tile bound here was measured in: _arith_dtype keeps bf16
    only for fp16/bf16 input under _DOT_TAPS_MAX taps, and in fp32 the x tile
    doubles against an accumulator budget that was already the binding one.

    Then whether it is worth it, which is two comparisons and both matter.  The
    run has to actually get longer -- see _pick_flat_blocks for what each form's
    run is -- and the compaction that buys has to stay small against the x
    traffic it saves.  Checking only the run is how (32,64,512) slipped through:
    its run gain is 1, so flat is pure loss there, and the loss is 1.07 GB of
    output.
    """
    if not use_dot or input.dtype not in (torch.float16, torch.bfloat16):
        return False
    N, C, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    if KH * KW > _DOT_TAPS_MAX or OC // groups < _DOT_MIN:
        return False
    # A 1-D shape lifted to H==1 never wins on flat: its blocked run is already
    # the whole W row (and a depthwise one additionally pays a densify), so the
    # flat identity only adds the compaction.  Measured, [2] (16,32,512)/d2 and
    # [10] (16,32,1024)/g32 are 3.3x and 2.4x *slower* on flat.
    if H == 1:
        return False
    PH, PW = padding
    wp = W + 2 * PW
    tot_p = oh * wp
    _, block_m = _pick_flat_blocks(OC // groups, tot_p)
    if block_m < _FLAT_RUN_GAIN * min(ow, _BLOCK_W_MAX):
        return False
    if wp != ow:
        moved = 2 * (N * OC * oh * ow)
        read = KH * KW * (N * C * (H + 2 * PH) * wp)
        # A C_in below _DOT_MIN is forced onto the padded-dot path, whose x read
        # is a C<16 strided gather at ~114 GB/s -- far below the flat byte-rate
        # the ratio above assumes.  The byte count understates that read's cost
        # and overstates the compaction's share, so admit a higher ratio there.
        # [20] (8,3,224,224)/k3 sits at 1.16 and flat wins 1.60x on it; the C_in
        # >= _DOT_MIN shapes that would also pass at the widened bound are all
        # 1-D and already refused above, so the tight 0.5 stays for them.
        cap = 8.0 if weight_c < _DOT_MIN else _FLAT_COMPACT_MAX
        if moved > cap * read:
            return False
    return True


def _flat_conv2d(input, weight, padding, dilation, groups, arith):
    N, C, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    PH, PW = padding
    DH, DW = dilation
    OH = H + 2 * PH - DH * (KH - 1)
    OW = W + 2 * PW - DW * (KW - 1)
    Hp, Wp = H + 2 * PH, W + 2 * PW
    c_stride = Hp * Wp
    tot_p = OH * Wp

    block_oc, block_m = _pick_flat_blocks(OC // groups, tot_p)
    block_c = min(_BLOCK_C_MAX, triton.next_power_of_2(weight_c))
    n_p = triton.cdiv(tot_p, block_m)
    # The channel tile is the only thing that can overshoot C_IN, and only its
    # last block: the weight is masked to zero there, so the garbage x lanes it
    # multiplies are harmless -- but their addresses still have to exist.
    c_eff = triton.cdiv(weight_c, block_c) * block_c

    tail = (
        (c_eff - weight_c) * c_stride
        + n_p * block_m
        + (KH - 1) * DH * Wp
        + (KW - 1) * DW
        + 1
    )
    src = _pad_input_flat(input.to(arith), padding, tail)

    out_plane = OH * Wp
    out_numel = N * OC * out_plane
    out_buf = torch.empty(
        out_numel + max(_SLACK_ELEMS, n_p * block_m),
        device=input.device,
        dtype=input.dtype,
    )
    wt = _prep_weight(weight, arith)
    grid = (n_p, triton.cdiv(OC // groups, block_oc), N * groups)
    _flat_conv2d_kernel[grid](
        src,
        wt,
        out_buf,
        groups * weight_c * c_stride,
        c_stride,
        OC * out_plane,
        out_plane,
        tot_p,
        Wp,
        weight_c,
        OC,
        KH=KH,
        KW=KW,
        DH=DH,
        DW=DW,
        GROUPS=groups,
        BLOCK_OC=block_oc,
        BLOCK_C=block_c,
        BLOCK_M=block_m,
        NEED_OCMASK=(OC // groups) % block_oc != 0,
        NEED_CMASK=weight_c % block_c != 0,
        num_warps=_NUM_WARPS,
    )
    if Wp == OW:
        # k1, or d==0: the flat plane *is* the output plane, no compaction.
        return out_buf[:out_numel].view(N, OC, OH, OW)

    rows = N * OC * OH
    out = torch.empty(N * OC * OH * OW, device=input.device, dtype=input.dtype)
    bw = _compact_block_w(OW)
    cgrid = (triton.cdiv(OW, bw), triton.cdiv(rows, _COMPACT_BLOCK_R))
    _compact_plane_kernel[cgrid](
        out_buf,
        out,
        OW,
        rows,
        WP=Wp,
        BLOCK_W=_compact_block_w(OW),
        BLOCK_R=_COMPACT_BLOCK_R,
        num_warps=_NUM_WARPS,
    )
    return out.view(N, OC, OH, OW)


def _use_flat_3d(input, weight, padding, stride, groups, od, oh, ow):
    """Is the flat 3-D kernel both legal and worth its compaction on this shape?

    Same two questions as _use_flat_2d, and the same answers.  Stride 1 in all
    three axes, because the constant tap offset is the flat image of
    ``oh*SH + kh*DH == (oh+kh)*DH``.  A run that actually gets longer, and a
    compaction that stays small against the x traffic it saves.

    The dtype clause the 2-D gate carries is not here: the 3-D launcher stays in
    fp32 whatever the input is, so there is no ``_arith_dtype`` arm to be on the
    wrong side of.  See _flat_conv3d for why.
    """
    N, C, D, H, W = input.shape
    OC, weight_c, _, KH, KW = weight.shape
    PD, PH, PW = padding
    SD, SH, SW = stride
    if SD != 1 or SH != 1 or SW != 1:
        return False
    if OC // groups < 1:
        return False
    Hp, Wp = H + 2 * PH, W + 2 * PW
    tot_p = oh * Wp
    _, block_m = _pick_flat_blocks(OC // groups, tot_p, floor_oc=_DOT_MIN)
    if block_m < _FLAT_RUN_GAIN * min(ow, _BLOCK_W_MAX):
        return False
    if Wp != ow:
        moved = 2 * (N * OC * od * oh * ow)
        read = KH * KW * (N * C * (D + 2 * PD) * Hp * Wp)
        if moved > _FLAT_COMPACT_MAX * read:
            return False
    return True


def _flat_conv3d(input, weight, padding, dilation, groups):
    """The flat kernel on a 3-D shape; see _flat_conv3d_kernel.

    fp32 unconditionally, where the 2-D launcher lets the native dtype through
    on the dot path.  That is the note in _direct_conv3d's own words: the 3-D
    dot shapes sat at ``tl.dot``'s minimum tile before this, and handing them
    bf16 measured 2.3-3.0x *slower*.  Flat changes the tile -- BLOCK_M is now
    the whole (oh, ow) plane, so the cube is operand-bound rather than latency-
    bound -- but the measurement that says bf16 is safe here has not been made,
    and the upcast on these volumes is a few hundred KB against a kernel that
    was 6x its own traffic.  Left as the blocked path had it.

    The output is one ``(OD, OH, Wp)`` slab per channel, compacted back to OW by
    _compact_plane_kernel.  That kernel wants a plain ``(rows, Wp)`` plane and
    gets one: within a channel the depth slices are contiguous, so
    ``(n, oc, od, oh)`` flattens to a row index with stride Wp on its own.
    """
    N, C, D, H, W = input.shape
    OC, weight_c, KD, KH, KW = weight.shape
    PD, PH, PW = padding
    DD, DH, DW = dilation
    Dp, Hp, Wp = D + 2 * PD, H + 2 * PH, W + 2 * PW
    OD = Dp - DD * (KD - 1)
    OH = Hp - DH * (KH - 1)
    OW = Wp - DW * (KW - 1)
    c_stride = Dp * Hp * Wp
    h_stride = Hp * Wp
    tot_p = OH * Wp

    block_oc, block_m = _pick_flat_blocks(OC // groups, tot_p, floor_oc=_DOT_MIN)
    # Not _pick_block_c: its _UB_TILE_MAX shrink is the *blocked* kernel's
    # budget, where BLOCK_C * BLOCK_W and the accumulator are spent together.
    # Here the accumulator is BLOCK_OC * BLOCK_M and the x tile rides alongside
    # it, exactly as in the 2-D flat kernel, so the only floor that matters is
    # tl.dot's -- needed for the c_in = 4 shape, whose padded tile is 16.
    block_c = max(_DOT_MIN, min(_BLOCK_C_MAX, triton.next_power_of_2(weight_c)))
    n_p = triton.cdiv(tot_p, block_m)
    c_eff = triton.cdiv(weight_c, block_c) * block_c
    tail = (
        (c_eff - weight_c) * c_stride
        + n_p * block_m
        + (KD - 1) * DD * h_stride
        + (KH - 1) * DH * Wp
        + (KW - 1) * DW
        + 1
    )
    src = _pad_input_flat(input.float(), padding, tail)

    out_plane = OH * Wp
    out_c_stride = OD * out_plane
    out_numel = N * OC * out_c_stride
    out_buf = torch.empty(
        out_numel + max(_SLACK_ELEMS, n_p * block_m),
        device=input.device,
        dtype=input.dtype,
    )
    wt = _prep_weight(weight, torch.float32)
    grid = (n_p, triton.cdiv(OC // groups, block_oc), N * groups * OD)
    _flat_conv3d_kernel[grid](
        src,
        wt,
        out_buf,
        groups * weight_c * c_stride,
        c_stride,
        h_stride,
        OC * out_c_stride,
        out_c_stride,
        out_plane,
        tot_p,
        Wp,
        weight_c,
        OC,
        KD=KD,
        KH=KH,
        KW=KW,
        DD=DD,
        DH=DH,
        DW=DW,
        GROUPS=groups,
        OD=OD,
        BLOCK_OC=block_oc,
        BLOCK_C=block_c,
        BLOCK_M=block_m,
        NEED_OCMASK=(OC // groups) % block_oc != 0,
        NEED_CMASK=weight_c % block_c != 0,
        num_warps=_NUM_WARPS,
    )
    if Wp == OW:
        return out_buf[:out_numel].view(N, OC, OD, OH, OW)

    rows = N * OC * OD * OH
    out = torch.empty(N * OC * OD * OH * OW, device=input.device, dtype=input.dtype)
    bw = _compact_block_w(OW)
    _compact_plane_kernel[(triton.cdiv(OW, bw), triton.cdiv(rows, _COMPACT_BLOCK_R))](
        out_buf,
        out,
        OW,
        rows,
        WP=Wp,
        BLOCK_W=bw,
        BLOCK_R=_COMPACT_BLOCK_R,
        num_warps=_NUM_WARPS,
    )
    return out.view(N, OC, OD, OH, OW)


def _direct_conv2d(input, weight, padding, stride, dilation, groups):
    N, _, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    block_oc, block_w = _pick_blocks(OW, OC // groups)
    use_dot = _can_use_dot(block_oc, block_w, weight_c)

    # _can_use_dot refuses a contraction axis below _DOT_MIN//2 (c_in < 8)
    # because padding it to 16 triples the dot's arithmetic, and in bf16/fp16
    # that is a 1.6x loss.  In fp32 the cube is far enough ahead of the vector
    # unit that the same pad is a win -- 406 us vs 560 us FMA on the suite's
    # only qualifying shape, [4] (8,3,224,224)/k3.  Force the dot path when the
    # output tile is already dot-sized and let _arith_dtype's fp32 override below
    # (weight_c < _DOT_MIN) pay for the padded contraction.
    if not use_dot and block_oc >= _DOT_MIN and block_w >= _DOT_MIN:
        use_dot = True

    # Depthwise is the one grouped shape the per-group tile cannot carry: with
    # one input and one output channel per group, BLOCK_OC is 1 and the tile
    # never reaches tl.dot.  Densifying the weight puts the same arithmetic on
    # the cube; see _densify_depthwise for what that measures.
    if not use_dot and groups > 1 and OC // groups == 1 and weight_c == 1:
        dense = _densify_depthwise(weight, input.shape[1])
        dense_oc, _ = _pick_blocks(OW, OC)
        if _can_use_dot(dense_oc, block_w, dense.shape[1]):
            return _direct_conv2d(input, dense, padding, stride, dilation, 1)

    block_c = _pick_block_c(weight_c, use_dot, block_w)
    # Large kernels walk their taps in a runtime loop (one dot in the unrolled
    # body) so the code stays bf16-capable without the unrolled-bf16 hang; see
    # _arith_dtype and the RUNTIME_TAPS arm of _direct_conv2d_kernel.  Whether
    # the loop is allowed to keep bf16 is decided after split_w below: the
    # split-width path re-derives its tap addresses through residue arithmetic
    # in the loop, and there the smaller operands are a net loss.
    runtime_taps = use_dot and KH * KW > _DOT_TAPS_MAX
    split_w = SW >= _SPLIT_MIN_STRIDE
    arith = _arith_dtype(
        input, use_dot, KH * KW, runtime_taps=runtime_taps and not split_w
    )
    # A dot forced onto a sub-16 contraction (the c_in < 8 arm above) pads its
    # channel tile to _DOT_MIN, and that padding is only worth it in fp32 -- the
    # native dtype would hand the cube 16/3 the operands for no gain.  _arith_dtype
    # would otherwise keep bf16 here (taps <= _DOT_TAPS_MAX), so pin it down.
    if use_dot and weight_c < _DOT_MIN:
        arith = torch.float32

    # Stride-1 shapes index the plane flat: one tap offset becomes a constant
    # shift, so the x tile's innermost run is BLOCK_M instead of BLOCK_W.  It is
    # a separate kernel because that run length is what the blocked form cannot
    # reach, and it costs a compaction of the Wp-strided result back to OW
    # columns.  See _flat_conv2d_kernel for the measurements and _use_flat_2d
    # for when it is not worth taking.
    if (
        SH == 1
        and SW == 1
        and _use_flat_2d(input, weight, padding, groups, OH, OW, use_dot)
    ):
        return _flat_conv2d(input, weight, padding, dilation, groups, arith)

    # The halo is only built when the padding is non-zero, which is the only
    # thing that can put a tap outside the input.  A row that is not a whole
    # number of tiles used to need one too, to give the last segment's masked
    # lanes somewhere to point; those lanes now carry ``w_ok`` on the load and
    # never form the address, so the zero-padded case reads the input directly.
    #
    # The upcast happens here rather than inside the pad.  ``arith`` is fp32
    # except on the dot path, where the native dtype is kept; see _arith_dtype.
    # A width stride turns every tap load into a stride-SW run; the split halo
    # is what turns it back into a contiguous one, and it needs the copy whether
    # or not there is padding to write.  See _pad_split_width.
    #
    # It is applied at every stride above 1, and the split on/off pair is
    # measured rather than assumed.  Same chip, same session, split against the
    # plain halo, gem device time in us:
    #
    #   [ 8] (8,8,8192)      k11 s4  1481  ->   149     10x
    #   [12] (32,64,210,210)  k5 s2  10759 -> 10021    1.07x
    #   [13] (16,32,24,24)    k3 s2   333  ->   321     1.04x  (noisy, 221-356)
    #   [ 7] (16,24,2048)     k7 s2   619  ->   616     1.00x
    #
    # so stride 4 is where it pays an order of magnitude and stride 2 is where
    # it is worth a few percent -- and re-measured on the whole call, stride 2
    # is still worth taking.  [12] without the split is 10.98 ms against 9.22
    # with it, so the conv's stride-2 penalty (5.2 ms there) exceeds the
    # transposing copy that buys it back (3.5 ms).  The clause above is right as
    # written: never a loss.
    # A 1x1 kernel has nothing to permute, and _prep_weight exists to *be* that
    # permutation rather than to cast.  Its destination is (T, C, OC) because
    # that is the layout whose innermost axis is the one the kernel walks per
    # tap; at T == 1 the source's own (OC, C) already has a unit-stride axis of
    # the same length, so the tile the kernel builds -- ``w[oc, c]`` at
    # ``oc * W_OC_STRIDE + c * w_c_stride`` -- is one contiguous run of
    # BLOCK_OC * BLOCK_C elements read straight out of the caller's tensor.
    # Written into the prepared layout it is BLOCK_C runs of BLOCK_OC, which is
    # the same bytes in a worse shape.  There is no gather to avoid, because
    # what makes the native layout a gather is the *tap* axis -- stride KH*KW
    # in the innermost one -- and at 1x1 that stride is 1.
    #
    # So the permutation is skipped, and with it a kernel launch.  That is the
    # whole win and it is not a small one: the launch is a fixed 2.6 us against
    # the 3.7 us the convolution itself takes on (4,16,256)/k1, so it is 41% of
    # that call and takes its speedup from 3.32x to 5.68x -- +0.107 on the
    # suite's arithmetic mean from one shape, where the whole 1-D halo work
    # below is worth +0.117 across seven.
    #
    # Gated on the dot arm because the FMA arm reads its weight as a vector over
    # oc, ``tap_w + c * w_c_stride``, and there the unit-stride axis is the one
    # the permutation creates: indexing that vector by oc through a native
    # (OC, C) means stride C, a gather per tap per channel.  The masks and the
    # tile divisibility are required, not preferred: without them a masked lane
    # computes an address past the end of the *caller's* tensor, which has none
    # of the slack _prep_weight's own buffer carries -- the mask suppresses the
    # access, not the address, and the MTE faults on the address first (see
    # _slack).  With both divisions exact every lane of the tile is in bounds,
    # masked or not.
    native_w = (
        use_dot
        and KH == 1
        and KW == 1
        and groups == 1
        and arith == weight.dtype
        and weight.is_contiguous()
        and weight_c % block_c == 0
        and (OC // groups) % block_oc == 0
    )
    # How far the last segment's masked lanes run past the padded row; the pad
    # widens the row by that much so their addresses stay inside it.
    tail = (
        (triton.cdiv(OW, block_w) * block_w - 1) * SW + (KW - 1) * DW + 1 - (W + 2 * PW)
    )
    # The weight permutation rides along with the 1-D halo when there is one to
    # ride along with, and is built on its own otherwise.  Both entries land in
    # `wt`; only a shape that reached the flat branch can set it here, and only
    # a shape that still needs a permutation is offered one -- handing the
    # native arm a weight would buy nothing and still pay for the kernel.
    # The upcast is not a launch of its own when the halo kernel is going to
    # write every element anyway.  It costs one there -- a wider store -- where
    # ``input.to(arith)`` is an aclnnInplaceCopy_CastAiCore of its own: 1.8 us
    # on [11] against a 9.4 us call and 3.4 us on [7] against 53.1 us, which is
    # the largest single line in [11]'s profile after the convolution.
    #
    # Only the contiguous case folds.  The kernel reads its source as
    # ``r*W + c``, and the one thing that lets a strided input reach it today is
    # that ``.to(arith)`` left a contiguous tensor behind on the way; a strided
    # input keeps that cast, since a source the kernel cannot index is worth
    # more than the launch it saves.
    cast_in = arith != input.dtype and input.is_contiguous()
    wt = None
    if split_w:
        src, wt = _pad_split_cast(
            input, padding, SW, DW, KW, OW, block_w, arith, None if native_w else weight
        )
    else:
        src, wt = _pad_flat_row(
            input if cast_in else input.to(arith),
            padding,
            tail,
            None if native_w else weight,
            arith,
        )
        if src is None:
            src = _pad_input(input.to(arith), padding, tail)
    in_s = src.stride()
    # The split tensor is (N, C, H, SW, Wq) and the plain one (N, C, H, W); the
    # kernel takes the residue-plane stride separately so both reach the same
    # expression.  Unused, and so zero, when there is no split.
    in_r_s, in_w_s = (in_s[3], in_s[4]) if split_w else (0, in_s[3])

    out_numel = N * OC * OH * OW
    out_buf = torch.empty(
        out_numel + _slack(OH * OW, block_oc, block_w),
        device=input.device,
        dtype=input.dtype,
    )
    output = out_buf[:out_numel].view(N, OC, OH, OW)
    # `wt` was set by the pad above when the permutation rode along with it, and
    # by the native arm when there was no permutation to build at all.  Falling
    # through both means it still has to be built, in its own launch.
    if native_w:
        wt = weight
        w_oc_stride = weight_c
    elif wt is None:
        wt = _prep_weight(weight, arith)
        w_oc_stride = 1
    else:
        w_oc_stride = 1
    out_s = output.stride()

    grid = (
        N * OH * triton.cdiv(OW, block_w),
        triton.cdiv(OC // groups, block_oc),
        groups,
    )
    _direct_conv2d_kernel[grid](
        src,
        wt,
        output,
        N,
        src.shape[2],
        src.shape[3],
        OC,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_r_s,
        in_w_s,
        # Stride of the C axis, which sits at -2 in the transposed (*K, C, OC)
        # layout.  Not stride(1): that is the kernel-height axis.
        wt.stride(weight.dim() - 2),
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        KH,
        KW,
        SH,
        SW,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
        BLOCK_OC=block_oc,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        NEED_CMASK=use_dot and weight_c % block_c != 0,
        USE_DOT=use_dot,
        W_SPLIT=split_w,
        RUNTIME_TAPS=runtime_taps,
        W_OC_STRIDE=w_oc_stride,
        num_warps=_NUM_WARPS,
    )
    return output


def _direct_conv3d(input, weight, padding, stride, dilation, groups):
    N, _, D, H, W = input.shape
    OC, weight_c, KD, KH, KW = weight.shape
    PD, PH, PW = padding
    SD, SH, SW = stride
    DD, DH, DW = dilation
    OD = _output_size(D, KD, SD, PD, DD)
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    block_oc, block_w = _pick_blocks(OW, OC // groups)
    dot_ok = _can_use_dot(block_oc, block_w, weight_c)

    # Widening BLOCK_W past a narrow OW to reach the dot arm was tried here and
    # is a loss: on (2,16,16,16,16)/k3/s2 -- the suite's only 3-D shape with
    # OW < _DOT_MIN, where BLOCK_OC is already the whole channel axis -- the
    # dot arm with a padded 16-wide tile runs the convolution in 219 us against
    # the FMA arm's 166.  These tiles sit at tl.dot's minimum, the runtime tap
    # loop gives the cube nothing to pipeline, and 27 dots of (16,16)x(16,16)
    # come out latency-bound exactly as _DOT_3D describes; the FMA arm's 432
    # rank-one updates at least keep the vector unit issuing.

    # Same depthwise lift as the 2D launcher and for the same reason; see
    # _densify_depthwise.  The 3D variant was missing it, which is why
    # (2,16,12,12,12)/3x3x3/g16 sat at 1380 us against torch's 71 us: BLOCK_OC
    # pins to 1 and the FMA kernel then does 27 rank-1 updates per output.
    #
    # The lift is decided on what the *dense* weight can reach, and the
    # recursion it starts takes the same arm a dense 3-D shape would.  With
    # _DOT_3D on that is the dot arm: the lift alone took this shape from 1380 us
    # to 301, and the arm from there to 177.  The lift itself still costs nothing
    # beyond the weight it builds.
    if not dot_ok and groups > 1 and OC // groups == 1 and weight_c == 1:
        dense = _densify_depthwise(weight, input.shape[1])
        dense_oc, _ = _pick_blocks(OW, OC)
        if _can_use_dot(dense_oc, block_w, dense.shape[1]):
            return _direct_conv3d(input, dense, padding, stride, dilation, 1)

    use_dot = dot_ok and _DOT_3D
    block_c = _pick_block_c(weight_c, use_dot, block_w)

    # Stride-1 shapes index the (oh, ow) plane flat, one depth slice per program;
    # see _flat_conv3d_kernel.  It is the 2-D launcher's move one dimension out,
    # and it is taken before the halo because it builds a different one -- the
    # un-widened rows the flat identity needs, not the widened rows the tile
    # loop needs.
    if _use_flat_3d(input, weight, padding, stride, groups, OD, OH, OW):
        return _flat_conv3d(input, weight, padding, dilation, groups)

    # Same halo rule as the 2D launcher; see the note there.  The dtype is fp32
    # unconditionally -- not gated on use_dot like the 2D launcher -- and that
    # is re-measured, not an oversight.  The dot arm walks its taps in a runtime
    # loop (see _DOT_3D), and a bf16 dot inside a runtime loop does not pipeline
    # the way the 2D kernel's unrolled bf16 dots do: handing the dot arm its
    # native bf16 tiles made [5]/[19]/[21] *slower* by 2.4x/3.0x/2.3x
    # (212 -> 507 us, 751 -> 2272 us, 170 -> 385 us), because these tiles sit at
    # tl.dot's minimum size and the cube is latency-bound there, so the smaller
    # operands buy nothing and the bf16 loads cost the loop its pipeline.  The
    # FMA arm has no choice in the matter (see _arith_dtype), so every 3-D call
    # stays fp32.
    src = _pad_input(
        input.float(),
        padding,
        (triton.cdiv(OW, block_w) * block_w - 1) * SW
        + (KW - 1) * DW
        + 1
        - (W + 2 * PW),
    )
    in_s = src.stride()

    out_numel = N * OC * OD * OH * OW
    out_buf = torch.empty(
        out_numel + _slack(OD * OH * OW, block_oc, block_w),
        device=input.device,
        dtype=input.dtype,
    )
    output = out_buf[:out_numel].view(N, OC, OD, OH, OW)
    wt = _prep_weight(weight, torch.float32)
    out_s = output.stride()

    grid = (
        N * OD * OH * triton.cdiv(OW, block_w),
        triton.cdiv(OC // groups, block_oc),
        groups,
    )
    _direct_conv3d_kernel[grid](
        src,
        wt,
        output,
        N,
        src.shape[2],
        src.shape[3],
        src.shape[4],
        OC,
        OD,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        in_s[4],
        # Stride of the C axis in the transposed (*K, C, OC) layout; see the
        # 2D launcher.
        wt.stride(weight.dim() - 2),
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        out_s[4],
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        DD,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
        BLOCK_OC=block_oc,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        NEED_CMASK=use_dot and weight_c % block_c != 0,
        USE_DOT=use_dot,
        num_warps=_NUM_WARPS,
    )
    return output


@libentry()
@triton.jit
def _im2col_gemm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    C,
    OC,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_h_stride,
    in_w_stride,
    w_k_stride,
    w_n_stride,
    out_n_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One (n, oh, ow-chunk) row per program over the M axis; the output-channel
    # block is the N axis.  n and oh are scalars so the halo's padded-coordinate
    # index ``ih = oh * SH + kh * DH`` stays a scalar+offset rather than a
    # gather; ow is the contiguous axis.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_chunk = tl.cdiv(OW, BLOCK_M)
    ow_chunk = pid_m % num_chunk
    tmp = pid_m // num_chunk
    oh = tmp % OH
    n = tmp // OH

    ow = ow_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = ow < OW
    n_mask = n_off < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = KH*KW*C_in, walked in BLOCK_K chunks over the flat (tap, channel) axis,
    # k = tap*C + c.  This is the big-K contraction the cube is built for: one
    # dot carries several taps' worth of channels instead of the direct kernel's
    # one tap per BLOCK_C dot.
    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_ok = k_off < K
        c = k_off % C
        # Clamp the tail tap so a masked lane's address stays on the last real
        # tap; the mask zeroes the lane, but its address must stay in bounds (see
        # _slack).
        tap = tl.minimum(k_off // C, KH * KW - 1)
        kh = tap // KW
        kw = tap % KW

        ih = oh * SH + kh * DH
        iw = ow[:, None] * SW + kw[None, :] * DW
        a_ptrs = (
            input_ptr
            + n * in_n_stride
            + c[None, :] * in_c_stride
            + ih[None, :] * in_h_stride
            + iw * in_w_stride
        )
        a = tl.load(a_ptrs, mask=k_ok[None, :] & m_mask[:, None], other=0.0)

        b_ptrs = weight_ptr + k_off[:, None] * w_k_stride + n_off[None, :] * w_n_stride
        b = tl.load(b_ptrs, mask=k_ok[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc)

    out_ptrs = (
        output_ptr
        + n * out_n_stride
        + n_off[None, :] * out_c_stride
        + oh * out_h_stride
        + ow[:, None] * out_w_stride
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _im2col_gemm_conv2d(input, weight, padding, stride, dilation, groups):
    """Fused im2col + GEMM for 2-D conv, matching CANN's default strategy.

    Out[M, N] = im2col(X)[M, K] @ W[K, N], with M = N*OH*OW output positions,
    K = KH*KW*C_in, N = OC.  The im2col is implicit -- the GEMM's A-load gathers
    the window directly -- so no im2col matrix is ever materialised (the thing
    CANN gets for free from its Load3D instruction; triton cannot express that,
    so the expansion here is a real gather, but the contraction axis is the full
    KH*KW*C_in instead of the direct kernel's BLOCK_C).
    """
    N, C, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)
    K = KH * KW * C

    block_m = min(256, max(16, triton.next_power_of_2(OW)))
    block_n = min(64, max(16, triton.next_power_of_2(OC)))
    block_k = min(K, max(16, _GEMM_K_TAPS * triton.next_power_of_2(C)))

    arith = input.dtype

    # Halo in the input's own dtype: the GEMM keeps bf16/fp16 on the dot, so the
    # fp32 upcast the FMA arm needs is not done here.
    src = _pad_input(
        input,
        padding,
        (triton.cdiv(OW, block_m) * block_m - 1) * SW
        + (KW - 1) * DW
        + 1
        - (W + 2 * PW),
    )
    in_s = src.stride()

    # _prep_weight's (KH, KW, C, OC) is, flattened, exactly (K, OC) row-major:
    # w_k_stride is the C-axis stride and w_n_stride is 1.
    wt = _prep_weight(weight, arith)
    w_k_stride = wt.stride(weight.dim() - 2)
    w_n_stride = 1

    out_numel = N * OC * OH * OW
    out_buf = torch.empty(
        out_numel + _slack(OH * OW, block_n, block_m),
        device=input.device,
        dtype=input.dtype,
    )
    output = out_buf[:out_numel].view(N, OC, OH, OW)
    out_s = output.stride()

    grid = (N * OH * triton.cdiv(OW, block_m), triton.cdiv(OC, block_n))
    _im2col_gemm_kernel[grid](
        src,
        wt,
        output,
        C,
        OC,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        w_k_stride,
        w_n_stride,
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        KH=KH,
        KW=KW,
        SH=SH,
        SW=SW,
        DH=DH,
        DW=DW,
        K=K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=_NUM_WARPS,
    )
    return output


def _use_im2col_gemm(ndim, groups, dilation, kh, kw, c_in):
    """Whether a shape takes the im2col+GEMM path rather than the direct one.

    CANN's default is im2col+GEMM; it only steps aside for 1x1 (a pure GEMM the
    direct kernel already runs as one dot), dilated kernels (the window is not a
    contiguous block, which is what Direct is for), depthwise/grouped (block-
    diagonal, not a single GEMM), 3-D (not handled here yet), and a contraction
    under tl.dot's minimum (C_in < 16 leaves BLOCK_K short of the cube's tile).
    Stride does not disqualify -- the direct kernel pays a split-width transpose
    for stride >= 2 that im2col folds into the gather for free.
    """
    if not _USE_IM2COL_GEMM:
        return False
    if ndim != 2 or groups != 1:
        return False
    if any(d != 1 for d in dilation):
        return False
    if c_in < _DOT_MIN:
        return False
    return kh * kw > 1


def _direct_conv(input, weight, padding, stride, dilation, groups, ndim):
    if ndim == 1:
        # Lift to 2D with a leading unit height rather than a trailing unit
        # width: the width axis is the one the kernel vectorizes over, so a
        # unit *width* would leave a single live lane per program.
        input2 = input.unsqueeze(2)
        weight2 = weight.unsqueeze(2)
        pad2 = [0, padding[0]]
        str2 = [1, stride[0]]
        dil2 = [1, dilation[0]]
        if _use_im2col_gemm(
            2, groups, dil2, weight2.shape[2], weight2.shape[3], weight2.shape[1]
        ):
            return _im2col_gemm_conv2d(
                input2, weight2, pad2, str2, dil2, groups
            ).squeeze(2)
        return _direct_conv2d(input2, weight2, pad2, str2, dil2, groups).squeeze(2)
    if ndim == 2:
        if _use_im2col_gemm(
            2, groups, dilation, weight.shape[2], weight.shape[3], weight.shape[1]
        ):
            return _im2col_gemm_conv2d(input, weight, padding, stride, dilation, groups)
        return _direct_conv2d(input, weight, padding, stride, dilation, groups)
    return _direct_conv3d(input, weight, padding, stride, dilation, groups)


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """
    Ascend implementation of the bias-free cuDNN convolution.

    Dimensions, parameter normalization and the returned layout follow the
    shared implementation in ``flag_gems/ops/cudnn_convolution.py``; only the
    arithmetic and the tiling differ (see the note above the direct kernels).

    ``benchmark``, ``deterministic`` and ``allow_tf32`` are accepted for
    interface compatibility and do not select an algorithm.
    """
    logger.debug("GEMS_ASCEND CUDNN_CONVOLUTION")

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )
    padding = _to_list(padding, ndim)
    stride = _to_list(stride, ndim)
    dilation = _to_list(dilation, ndim)

    # Depthwise used to be handed to the shared kernel on the grounds that it
    # has no cross-channel reduction, so a tile with one output channel per
    # program and a scalar weight per tap would be cheaper than a (BLOCK_OC,
    # BLOCK_W) outer product that is 7/8 masked out.  That is true of the
    # *arithmetic* and false of the memory access: the shared tile addresses its
    # input through a gather per tap, which is the one thing this backend
    # charges for (see the note at the top of this file).  The direct path's row
    # tiling makes those loads contiguous instead, and depthwise is the shape it
    # suits best -- groups == C_in collapses BLOCK_OC to 1, so there is no
    # masked-out outer product left to pay for either way.
    #
    # Measured, wall clock, bf16, against the shared kernel on the same tensors
    # and the same torch reference:
    #
    #   (16,32,56,56)  k3 g=32   28.63 ms -> 4.43 ms   6.5x
    #   (16,32,1024)   k3 g=32    3.55 ms -> 0.45 ms   7.8x
    #   (2,16,12,12,12) k3 g=16   3.10 ms -> 1.82 ms   1.7x
    #
    # all three bit-exact against the shared path (max abs diff 0.0).  Same
    # lesson as the channel bound below: the comparison that picked the shared
    # path was made against the pre-row-tiling kernel and did not survive it.
    if input.dtype == torch.float32:
        return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)

    # --- fp16 / bf16 -------------------------------------------------------
    # A 1x1 kernel with one group, unit stride and no padding is a plain GEMM,
    # and it went to the shared pointwise kernel until this was re-measured, on
    # the grounds that that one keeps the tensor core.  It does, and so does the
    # direct path -- which on a single tap is ``tl.dot`` over the whole channel
    # reduction and nothing else -- and the direct path is faster on every shape
    # measured, bf16, gems device time in us, shared against direct:
    #
    #   (16,64,56,56)   k1 oc  64   139.5   71.4   1.95x
    #   (32,128,1024)   k1 oc 128    91.4   25.8   3.55x      <- the widest margin,
    #   (8,256,56,56)   k1 oc 256   139.1  118.0   1.18x         and the shape a
    #   (4,512,32,32)   k1 oc 512    58.3   48.3   1.21x         GEMM most suits
    #   (64,32,512)     k1 oc  32    13.3    8.9   1.50x
    #
    # The clause that picked it was written before the row tiling above and did
    # not survive it, the same way the channel bound below did not.
    #
    # Everything else: the direct kernels above, in the dtype each of them
    # asks for.  The kernels are handed the tensor as it arrived rather than an
    # upcast image of it; _direct_conv2d/_direct_conv3d decide per kernel, and
    # only the tl.dot branch keeps the native dtype (see _arith_dtype).  Handing
    # bf16 to a kernel that is *not* on the dot path is a large loss: the kernel
    # is unchanged but bf16 makes the 1D shape that runs in 1.06 ms in fp32 take
    # 87.5 ms, because the Ascend backend does not vectorize the in-loop bf16
    # load / .to(tl.float32).  Converting once up front costs a linear pass over
    # the operands and buys that 80x back, so those paths still do it.
    #
    # Folding that conversion into the halo's copy is the obvious saving and was
    # measured: no better on any case, 4.8% worse on one.  See _pad_input.
    #
    # It also fixes a correctness hole: the shared conv1d/2d/3d kernels
    # mis-compute 1x1 kernels in every dtype (measured 18.0 absolute error in
    # fp16 and 17.99 in fp32 against an fp64 reference).  The direct kernels
    # handle 1x1 fine, so every 1x1 reaches them now, padded or not, and
    # regardless of channel count.
    #
    # There is deliberately no channel bound on this.  There used to be one --
    # the direct path was capped at C_in <= 64, on the grounds that it walks
    # C_in * KH * KW taps per output where the implicit GEMM amortizes over a
    # larger K.  That was measured against the kernel that clamped every tap to
    # stay in bounds, and materialising the halo instead removed exactly the
    # per-tap cost the bound was reasoning about.  Re-measured on the current
    # kernel (bf16, 3x3, s1, oc == C_in, (16, C_in, 56, 56)):
    #
    #   C_in      8      16      32      64     256
    #   shared  10.5    16.0    32.8    89.1   902.6  ms
    #   direct   0.78    0.45    0.46    0.70    8.66  ms
    #   ratio    13x     35x     71x    127x    104x
    #
    # and at s2 / C_in=256, 81.7 ms against 6.14 ms (13x).  The margin grows
    # with C_in rather than shrinking -- the shared path's cost grows with it
    # while the direct path's stays flat -- so the old bound was not merely
    # stale, it was inverted: it handed the shapes with the most to gain to the
    # slower path.  Every shape measured is 12x-127x the direct path's cost,
    # which is well outside any noise that would make this a close call.
    return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)
