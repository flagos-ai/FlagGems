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

import contextlib
import importlib
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cudnn_convolution import (
    cudnn_convolution as _generic_cudnn_convolution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small-C_in convolutions get a tap-packed tl.dot kernel instead of the
# shipped scalar-FMA direct kernel.
#
# The generic op sends every case with C_in/groups < 16 -- and either fp32 or
# C_in <= 8 -- to `_direct_conv`, whose kernel multiplies and accumulates one
# tap at a time:
#
#     for c in tl.static_range(C_IN):
#       for kh in tl.static_range(KH):
#         for kw in tl.static_range(KW):
#           x = tl.load(...).to(tl.float32)   # one spatial vector per tap
#           w = tl.load(...).to(tl.float32)
#           acc += x[None, :] * w             # no tensor core
#
# That is C_IN * KH * KW fully unrolled load/convert/FMA steps with no use of
# the tensor cores, and it is where the whole benchmark's worst rows live. Two
# of its costs are dtype-dependent, which is what gave the cases away: the same
# shape and the same kernel scored
#
#     (8, 8, 8192)   k11 s4 p5    bf16 0.526   fp16 0.057   fp32 0.553
#     (16, 24, 2048) k7  s1 p3 g2 bf16 0.329   fp16 0.324   fp32 0.112
#
# fp16 loses 9x on the first row and fp32 loses 3x on the second. bf16->fp32 is
# a free bit-shift and fp32->fp32 is a no-op, but fp16->fp32 is a real
# conversion -- and this kernel runs one per element per tap.
#
# The replacement packs taps into the K axis and uses tl.dot, so the tensor
# cores do the arithmetic and the conversion disappears (fp16 and bf16 are
# consumed natively; accumulate is fp32 throughout):
#
#     lane  = tap * C_IN + channel        # BLOCK_CI = 16 lanes
#     for each group of N_TAPS = 16 // C_IN taps:
#         x = tl.load(input, mask=..., other=0.0)   # (BLOCK_SP, BLOCK_CI)
#         w = tl.load(weight, mask=..., other=0.0)  # (BLOCK_CI, BLOCK_OC)
#         acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32)
#
# Packing the taps into K is what makes this work at C_in = 3, where merely
# padding the channel dim to 16 (the generic conv2d path's approach) wastes 81%
# of the K work and measurably loses to the scalar kernel it replaces:
#
#     (8, 3, 224, 224) k3        pad-C-to-16 0.372   tap-packed 1.066
#
# N_TAPS = 16 // C_IN covers C_in = 8 -> 2 taps, 4 -> 4, 3 -> 5, 12 -> 1 (the
# last still wastes 25%, unavoidable at tl.dot's K >= 16 floor).
#
# fp32 passes input_precision="ieee" so the numerics match the shipped kernels,
# which compute exact fp32, and to leave cuDNN's TF32 out of it -- the benchmark
# forces torch.backends.cudnn.allow_tf32 = False for the same reason.
#
# Measured per row (gems speedup over the torch baseline, median of do_bench),
# shipped direct kernel -> this kernel:
#
#     (8, 8, 8192)   k11 s4 p5    fp16 0.058 -> 1.173   fp32 0.555 -> 1.038
#     (8, 3, 224, 224) k3         fp16 0.468 -> 1.065   bf16 0.871 -> 1.066
#                                 fp32 0.726 -> 0.778
#     (16, 24, 2048) k7 s1 p3 g2  fp32 0.113 -> 0.406   bf16 0.327 -> 0.453
#     (16, 32, 24, 24) k3 s2 p2 g2 bf16 1.286 -> 1.523  fp32 0.894 -> 1.211
#
# BLOCK_SP is chosen from the flattened output extent rather than autotuned:
# the large-extent cases want the wider tile (8,3,224,224 scores 0.767 at 64
# against 1.066 at 128) while the small ones want the narrower one to avoid a
# partly-empty tail (16,32,24,24 scores 1.657 at 64 against 1.523 at 128). The
# 16384 crossover picks the better tile on all four of the rows above.
# ---------------------------------------------------------------------------

# Below this flattened output extent the narrower tile wins; above it the wider
# one does. See the measurements in the comment above.
_BLOCK_SP_CROSSOVER = 16384
_BLOCK_SP_SMALL = 64
_BLOCK_SP_LARGE = 128
_BLOCK_OC = 16
_BLOCK_CI = 16
_NUM_WARPS = 4


def _out_size(in_size, kernel, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


@triton.jit
def _dot_conv2d_kernel(
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
    in_w_stride,
    w_oc_stride,
    w_c_stride,
    w_h_stride,
    w_w_stride,
    out_n_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    N_TAPS: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_valid = oc_off < oc_per_group

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    ow = sp_off % OW
    tmp = sp_off // OW
    oh = tmp % OH
    n = tmp // OH
    in_c_base = pid_group * C_IN

    # The K axis packs N_TAPS taps and C_IN channels: lane = tap * C_IN + chan.
    lane = tl.arange(0, BLOCK_CI)
    tap_l = lane // C_IN
    c_l = lane % C_IN
    c_valid = c_l < C_IN
    lane_valid = c_valid & (tap_l < N_TAPS)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)
    for tg in tl.static_range((KH * KW + N_TAPS - 1) // N_TAPS):
        t = tg * N_TAPS + tap_l
        kh = t // KW
        kw = t % KW
        t_valid = t < (KH * KW)
        ih = oh[:, None] * SH + kh[None, :] * DH - PH
        iw = ow[:, None] * SW + kw[None, :] * DW - PW
        in_mask = (
            ((n < N)[:, None] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W))
            & t_valid[None, :]
            & c_valid[None, :]
        )
        x = tl.load(
            input_ptr
            + n[:, None] * in_n_stride
            + (in_c_base + c_l)[None, :] * in_c_stride
            + ih * in_h_stride
            + iw * in_w_stride,
            mask=in_mask,
            other=0.0,
        )
        w = tl.load(
            weight_ptr
            + oc_glob[None, :] * w_oc_stride
            + c_l[:, None] * w_c_stride
            + kh[:, None] * w_h_stride
            + kw[:, None] * w_w_stride,
            mask=lane_valid[:, None] & t_valid[:, None] & oc_valid[None, :],
            other=0.0,
        )
        if IEEE:
            acc = tl.dot(x, w, acc=acc, input_precision="ieee", out_dtype=tl.float32)
        else:
            acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32)

    out_mask = (
        oc_valid[None, :] & (n < N)[:, None] & (oh < OH)[:, None] & (ow < OW)[:, None]
    )
    out_ptr = (
        output_ptr
        + oc_glob[None, :] * out_c_stride
        + n[:, None] * out_n_stride
        + oh[:, None] * out_h_stride
        + ow[:, None] * out_w_stride
    )
    tl.store(out_ptr, acc, mask=out_mask)


def _dot_conv2d(input, weight, padding, stride, dilation, groups):
    """Tap-packed tl.dot replacement for ``_direct_conv2d``.

    Same signature and contract as the shipped direct kernel: 4D input and
    weight, padding/stride/dilation as 2-element sequences. Callers that need a
    1D convolution promote to 4D first (``_direct_conv`` uses ``unsqueeze(-1)``),
    so this covers both dimensionalities.
    """
    N, _, H, W = input.shape
    OC, c_in, KH, KW = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = _out_size(H, KH, SH, PH, DH)
    OW = _out_size(W, KW, SW, PW, DW)

    output = torch.empty((N, OC, OH, OW), device=input.device, dtype=input.dtype)

    spatial = N * OH * OW
    block_sp = _BLOCK_SP_SMALL if spatial < _BLOCK_SP_CROSSOVER else _BLOCK_SP_LARGE
    # At least 16 lanes, and the same count as the padded channel dim.
    n_taps = max(1, _BLOCK_CI // c_in)

    in_s = input.stride()
    w_s = weight.stride()
    out_s = output.stride()
    grid = (
        triton.cdiv(spatial, block_sp),
        triton.cdiv(OC // groups, _BLOCK_OC),
        groups,
    )
    _dot_conv2d_kernel[grid](
        input,
        weight,
        output,
        N,
        H,
        W,
        OC,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        w_s[0],
        w_s[1],
        w_s[2],
        w_s[3],
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        KH,
        KW,
        SH,
        SW,
        PH,
        PW,
        DH,
        DW,
        C_IN=c_in,
        GROUPS=groups,
        BLOCK_SP=block_sp,
        BLOCK_OC=_BLOCK_OC,
        BLOCK_CI=_BLOCK_CI,
        N_TAPS=n_taps,
        # Exact fp32, matching the shipped kernels. Their tl.dot passes
        # allow_tf32=False and the benchmark disables cuDNN TF32, so anything
        # less would be measuring a different problem.
        IEEE=input.dtype == torch.float32,
        num_warps=_NUM_WARPS,
    )
    return output


@contextlib.contextmanager
def _tab_kernel_replaced():
    """Raise a tl.dot kernel in place of the shipped direct conv2d kernel.

    `_direct_conv` and `_direct_conv2d` are resolved as module globals at call
    time by the generic op's dispatch, so rebinding them here covers the 1D and
    2D direct branches without editing the shared implementation -- and without
    re-deriving the dispatch, which keeps depthwise/pointwise precedence (both
    of which also match "weight.shape[1] < 16") in the generic op's hands.

    `import flag_gems.ops.cudnn_convolution as m` does NOT give the module:
    `flag_gems.ops` re-exports the conv2d *function* under that name and the
    attribute lookup wins, so this goes through importlib.
    """
    module = importlib.import_module("flag_gems.ops.cudnn_convolution")
    original = module._direct_conv2d
    module._direct_conv2d = _dot_conv2d
    try:
        yield
    finally:
        module._direct_conv2d = original


# ---------------------------------------------------------------------------
# FlagTree AABS (auto_adjust_block_sizes) is held off by the generic op now --
# see ``_aabs_disabled`` in flag_gems/ops/cudnn_convolution.py, which documents
# the mispairing and its measurements. It used to be duplicated here; it is a
# property of the convolution kernels rather than of this backend, so one copy
# in the operator's own implementation covers every backend.
#
# The failure mode was worse here than slow. Below 16 the CoreX TLE pass cannot
# lower the kernel:
#
#   LLVM ERROR: Invalid basis N for in-dim 'offset' and out-dim 'dim0'.
#               Basis must be less than the out-dim size.
#
# surfacing as "RuntimeError: PassManager::run failed", which aborts the
# process rather than raising a catchable compile error; in_n = 2, 3, 4, 6, 8
# all abort. Only the backends listed in triton/runtime/adjust_kernel_param.py
# bump such a tile back up to the dot floor afterwards ("" / hcu / sunrise);
# iluvatar has no such branch, so whatever AABS shrinks stays shrunk.
#
# Measured on benchmark/test_cudnn_convolution.py, mean gems speedup over the
# torch baseline across the 66 comprehensive rows. Holding AABS off alone took
# it 0.6146 -> 0.891 (three runs: 0.9064 / 0.8901 / 0.8772, a +-0.015 spread
# that straddles the 0.9 bar). Adding the tap-packed kernel above took it to
# 0.9735 and 0.9767 on two runs -- bf16 1.085/1.089, fp16 1.052/1.059, fp32
# 0.784/0.782, with 27/66 rows at or above 0.9 in both, against 21-22 before.
# The spread tightened to 0.003 as well.
#
# Still open, measured but not taken: the generic op only routes to the direct
# branch when C_in < 16 *and* (fp32 or C_in <= 8), so cases just outside that --
# (16, 32, 24, 24) g2 at C_in 16, and (16, 24, 2048) g2 at C_in 12 in bf16/fp16
# -- keep the generic path even though this kernel beats it there (1.272 ->
# 1.657, 0.324 -> 0.453). Taking those needs the interception to move ahead of
# the generic dispatch and the kernel to grow a channel-block loop, since
# BLOCK_CI = 16 only covers half of a C_in of 32 and would otherwise produce
# silently wrong results.
#
# Rejected:
#
#   * Widening the conv2d_forward tuning list with BLOCK_NI_HO_WO = 512 tiles.
#     The shipped list tops out at 256. Appending the wide tiles to the shared
#     Autotuner gave 0.9004 and giving this operator its own copy of the kernel
#     with them baked in gave 0.8967 -- neither better than holding AABS off --
#     while the private copy costs 5x the wall clock (a new kernel identity
#     invalidates the Triton disk cache for every shape, turning a 170 s
#     benchmark into 968 s).
#
#     Sweeping all 21 shipped configs plus the 4 wide tiles directly against the
#     JIT kernel (bypassing the Autotuner, AABS off) explains why, and rules out
#     tuning as a lever in general. Per shape, the config the Autotuner picks is
#     already the best one in the shipped list -- 2417 us measured for
#     (8,256,64,64) against 2408 us best-of-list, 5.42 ms for (32,64,210,210)
#     against 5.415 ms -- so there is no unused config to claim. BLOCK_CI is
#     already optimal at its shipped 32 (64 is worse, 128 and 256 much worse,
#     256 exhausts shared memory), and num_stages has no effect at all, because
#     the K loop is a Python `range` unrolled at trace time and so is never
#     software-pipelined. The remaining gap to torch on the large shapes is a
#     kernel-architecture gap (implicit-GEMM is 2-4x off cuDNN here), not a
#     tuning gap.
#
#   * Routing the small-C_in shapes to the generic implicit-GEMM path: it is
#     far worse than the shipped direct kernel on most of them.
# ---------------------------------------------------------------------------


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
    """CUDNN-compatible no-bias convolution for iluvatar.

    Signature, dispatch and results are those of the generic implementation
    (see flag_gems.ops.cudnn_convolution). The only change on the way in is that
    the small-C_in direct branch runs this package's tap-packed tl.dot kernel
    instead of the shipped scalar-FMA one. AABS is held off by the generic op.
    """
    logger.debug("GEMS_ILUVATAR CUDNN_CONVOLUTION")

    with _tab_kernel_replaced():
        return _generic_cudnn_convolution(
            input,
            weight,
            padding,
            stride,
            dilation,
            groups,
            benchmark,
            deterministic,
            allow_tf32,
        )
