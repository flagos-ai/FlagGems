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
# Each program owns whole output rows, so the inner tl.arange is the output
# column index and the tap address ``iw = ow * SW + kw * DW`` is affine in it.
# The shared kernels tile the flattened (n, oh, ow) space instead, which breaks
# that affine form at every row boundary and turns each tap into a gather -- the
# one op this backend charges orders of magnitude for.  The padding is
# materialised (_pad_input) rather than clamped for the same reason.
#
# Each rank has two arms, a tl.dot one and a per-tap FMA one for tiles below
# tl.dot's minimum, selected per call by _can_use_dot and reading the transposed
# (KH, KW, C, OC) weight _prep_weight builds.
# ---------------------------------------------------------------------------

# Spatial tile first -- a whole output row, capped at _BLOCK_W_MAX -- and the
# channel tile from what is left of _BLOCK_ELEMS.  Picked per call because the
# kernel is launch- and mask-bound at the small end of the shape range.  BLOCK_W
# is never traded down for BLOCK_OC: it is the tap load's run length.
_BLOCK_W_MAX = 512
# Ceiling on the output-channel tile.  The bytes a program pulls per MAC are
# 2 / BLOCK_OC, so this is the throughput knob; it is set from the accumulator
# budget below rather than at a constant.
_BLOCK_OC_MAX = 256
# Target accumulator size, BLOCK_OC * BLOCK_W, in elements.  BLOCK_C gives up
# whatever BLOCK_W or BLOCK_OC take; the budget is conserved exactly.
_BLOCK_ELEMS = 32768
# Channel tile for the tl.dot path, in elements.  Caps how much of a very wide
# channel dim is reduced per dot.
_BLOCK_C_MAX = 64
# tl.dot needs every dimension at least this large; below it there is no dot.
_DOT_MIN = 16
# Ceiling on the tl.dot path's input tile (BLOCK_C * BLOCK_W), in elements, from
# the unified buffer.  A correctness bound and not a speed knob: past it the
# build fails with "ub overflow", or a tile that only just overruns compiles and
# then faults the device.  Shrink BLOCK_C rather than BLOCK_W.
_UB_TILE_MAX = 8192
# Tiles for _compact_plane_kernel, swept at fixed volume on the (rows, 132) ->
# (rows, 130) shape.  Every other form of the same copy is worse; see the kernel.
_COMPACT_BLOCK_W = 256
_COMPACT_BLOCK_R = 128


def _compact_block_w(ow):
    """Column tile for _compact_plane_kernel: the padded row, rounded up.

    The load is a masked tile of a row that is only OW wide, so a fixed
    256-column tile fetches far more than it keeps when OW is small.
    """
    return min(_COMPACT_BLOCK_W, triton.next_power_of_2(max(1, ow)))


# Unrolled taps a kernel may have and still be handed fp16/bf16 tiles for its
# dots.  Not a tuning knob: past it bf16 hangs the device.  See _arith_dtype.
_DOT_TAPS_MAX = 9
# Whether the 3-D kernel may take its tl.dot arm.  On, now that the arm's one
# bug is fixed -- its weight tile's channel-reduction offset was missing the
# ``* w_c_stride`` the 2-D form carries -- and its error is at the same noise
# floor as every other dot path here.  The two 3-D shapes it does not move never
# reach it: stride 2 leaves an 8-wide row and C_in 4 is a 4-deep contraction,
# both under tl.dot's minimum.
_DOT_3D = True
# Cubes per program; named rather than a literal at the launch site so it can be
# swept.
_NUM_WARPS = 4
# Whether the 2-D non-depthwise, non-dilated shapes take im2col+GEMM (see
# _im2col_gemm_conv2d) -- the "match CANN's default strategy" move.  Off: CANN
# gets its big-K contraction from its Load3D hardware move, which triton cannot
# express, so the expansion here is a per-lane gather plus an emulated division
# on a runtime C -- exactly the two ops this backend charges ~4 orders of
# magnitude for -- and im2col measured 83x-2320x slower than the direct kernel.
_USE_IM2COL_GEMM = False
# Reduction tile of the im2col GEMM, in whole input channels: BLOCK_K =
# _GEMM_K_TAPS * next_power_of_2(C_in).  Larger = longer dots.
_GEMM_K_TAPS = 1
# Smallest width stride for which the split halo is built: a fixed cost against a
# shrinking benefit, so the crossover is measured per shape.  See
# _pad_split_width.
_SPLIT_MIN_STRIDE = 2

# Innermost halo run, in bytes, at which a strided ``copy_`` is still worth its
# one pass over the interior.  Above one cache line the run is contiguous and
# moves at full bandwidth; below it it degenerates into a per-element gather.
_PAD_RUN_BYTES = 128
# Elements each program of the flat-row halo kernel covers, used to derive its
# BLOCK_R from the row width.  A count and not a row count because the row width
# is what varies by 16x across the 1-D suite.  Swept on the whole call; re-sweep
# it whenever the flat branch's kernel changes.
_PAD_FLAT_ELEMS = 4096
# Spare elements left at the end of every buffer the kernels index directly, so
# that a masked lane's address stays inside the allocation.  See _slack.
_SLACK_ELEMS = 8192
# Side of the square tile the two weight-stage kernels use.  Sized for the load
# rather than the grid; see _prep_blocks.
_PREP_BLOCK = 64
# Tiles for the fused split halo: rows of the (n, c, hh) index space by columns
# of the split width axis.  ROWS must divide BLOCK_ROWS or the affine row ids
# overshoot the halo -- there is no clamp to fall back on; see the kernel.
_SPLIT_BLOCK_ROWS = 64
_SPLIT_BLOCK_Q = 128
# Upper bound on BLOCK_ROWS * next_power_of_2(W) for _fused_split_cast_kernel,
# which loads a whole input row at once rather than tiling the width: the tile
# must fit the unified buffer, so the wide rows stay on the tiled kernel below.
_FUSE_SPLIT_MAX_TILE = 16384
# Width tile (input columns) for _fused_split_cast_1d_kernel, the 1-D
# width-tiled sibling of _fused_split_cast_kernel.  Same unified-buffer budget as
# the whole-row load.
_FUSE_SPLIT_BLOCK_W = 256


def _arith_dtype(input, use_dot, taps, runtime_taps=False):
    """The dtype the direct kernels should do their arithmetic in.

    fp32, except that a small-tap kernel taking the ``tl.dot`` branch can keep
    fp16/bf16: the products are exact either way and the upcast is a pass over each
    operand.

    ``taps`` is a correctness bound: a bf16 dot past _DOT_TAPS_MAX unrolled taps hangs
    the *device*.  The FMA branch must stay fp32 regardless -- the backend does not
    vectorize a bf16 load feeding a multiply.  ``runtime_taps`` selects the runtime
    tap loop, passed only when ``not split_w``.
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
    # Zero the padded buffer in one flat pass; the interior copy below overwrites
    # the middle.  Splitting fill from copy is what keeps every load address
    # affine -- no tap clamp, no halo mask on the load.
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

    Memset tiles take ``[0, N_MEMSET)`` and the weight tiles the rest, so the split
    halo's fill carries the weight permutation and the prep's launch is saved.  The
    branch must be real so the else-arm's ``pid - N_MEMSET`` stays non-negative.
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
    # Copy the unpadded interior into its place in the padded buffer.  Plane id,
    # r and c all come out of the grid or an arange, so load and store are affine
    # and the only mask is the block overshoot.
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

    The interior is a row of ``W`` read from the source's own row stride and written
    ``PW`` further on; the borders are masked zero stores off the same rows.  Every
    address is affine and in bounds with no clamp, and W must be a power of two so
    ``tl.arange`` spans a whole row.

    ``CAST_FP32`` upcasts a bf16/fp16 source and writes fp32 out, folding away the
    launch ``input.to(arith)`` would have spent.
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

    Pad tiles take ``[0, N_PAD)`` and the weight tiles the rest, buying a launch.  The
    branch must be real: if the backend ran both arms of every program the else-arm's
    ``pid - N_PAD`` would index negatively and fault the MTE before any mask was
    consulted.

    ``GRID1`` folds the prep kernel's second axis in, since a flat grid has one axis:
    ``q = pid - N_PAD`` is that kernel's own ``p0 * GRID1 + p1``.
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
        # Named apart from the pad arm's ``v``: the two branches merge, so a name
        # assigned on both sides has to have one type, and these are
        # (BLOCK_CT, BLOCK_OC) against (BLOCK_R, W).
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
    # As the 2-D interior copy, but a (H, W) tile per depth plane rather than one
    # program per row -- a one-row grid is the anti-pattern _pad_split_width
    # measures.  d/h/w are all plain grid/arange ids, so every address is affine.
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
    # Zero a horizontal band of full-width rows starting at row_base.
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
    # rows, offset PH.
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

    Returns ``(src, wt)``; ``(None, None)`` when the shape does not reach here, and
    ``(src, None)`` when it does but no weight was handed over.

    The 1-D caller is lifted to H == 1, PH == 0, so (n, c, h) collapses into a flat
    axis of N*C rows; left alone it would take aclnn's ViewCopy, which issues N*C
    short strided rows.  The row axis must divide BLOCK_R and W must be a power of
    two, which lets the load run with neither a mask nor slack.  ``weight`` folds the
    _prep_weight permutation in; ``out_dtype`` upcasts a bf16/fp16 source here rather
    than in a launch of its own.
    """
    # Early-out: no padding and no tail means no halo, and the input is already a valid
    # source.  The launcher calls this directly, so it makes that call itself.
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

    The kernels use unpadded tap indices (``oh * SH + kh * DH``), so a tap in the halo
    must read a real zero.  That is what lets every tap load run unmasked and keep its
    address affine: the equivalent clamp takes the address out of the affine form the
    backend's axis analysis needs, for orders of magnitude, and masking the load
    instead costs more than the copy it saves.

    ``tail`` is how far past a padded row the last output tile's masked lanes run;
    widening the row by it gives their addresses room inside the tensor (see _slack).
    Callers pass fp32 even for fp16/bf16 inputs, so the halo is fp32.
    """
    tail = max(0, tail)
    if not any(padding) and tail == 0:
        return input
    # The 1-D lift, before either branch below, because both are the wrong shape
    # for it.  Shapes it does not take fall through unchanged.
    flat = _pad_flat_row(input, padding, tail)
    if flat[0] is not None:
        return flat[0]
    # Wide innermost run: the halo is a ``torch.zeros`` + strided ``copy_``, which
    # moves every byte once at full bandwidth.  A triton memset + interior-copy
    # pair is slower for the same interior, so only the narrow runs below use it.
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
    # an affine interior copy.  Both passes keep every address affine.
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

    A tap reads the input at ``ow * SW + kw * DW``, a stride-SW run.  The MTE pays for
    the cache lines a run touches rather than the bytes it keeps, so that costs SW
    times the traffic, and past stride 2 the backend stops treating it as a run at
    all.  Reshaping the padded width to ``(W/SW, SW)`` and transposing puts the same
    tap at plane ``(kw*DW) % SW`` and column ``ow + (kw*DW)//SW``: unit stride in
    ``ow``, with the plane a launch-time constant.  No element moves and no arithmetic
    changes, so the result is bit-identical.

    The halo itself is still built by _pad_input; this adds one transposing copy.
    """
    # No tail: the split layout has its own, appended flat past the last plane,
    # because the overshoot there leaves the row rather than the tensor.
    halo = _pad_input(input, padding, 0).contiguous()
    n, c = halo.shape[0], halo.shape[1]
    spatial = halo.shape[2:-1]
    wq = triton.cdiv(halo.shape[-1], sw)
    if wq * sw != halo.shape[-1]:
        # _pad_input pads symmetrically, so the split can need up to SW-1
        # columns of zeros on the right to make the planes whole.
        halo = _pad_input(halo, (0,) * (halo.ndim - 2), wq * sw - halo.shape[-1])
    # Splitting the width removes the row's overshoot room -- the overshoot is now
    # the plane's -- so the same allowance is appended past the last plane.
    # (The split can also need up to SW-1 columns of zeros on the right to make
    # the planes whole; _pad_input pads symmetrically.)
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = n * c * sw * wq
    for s in spatial:
        numel *= s
    # Not zeros: the transposing copy below writes every element of ``out``, and
    # the tail past ``numel`` is only ever addressed by lanes the kernel masks
    # off, so it needs room rather than a value.
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

    Replaces the three-kernel chain (Cast, PadV3, vendor Transpose) that built the
    split halo, all three issue-bound on their access pattern rather than bandwidth.

    The halo is read as a flat (ROWS, row_stride) tensor with plain ``tl.arange`` ids
    and no mask or clamp.  That is a hard requirement: ``reshape`` + ``permute``
    silently produce zero output whenever the tensor they fold came from a load whose
    index passed through any non-affine expression, a ``tl.minimum`` clamp even when
    it is a no-op.  ROWS must divide BLOCK_ROWS so these ids stay in bounds.  The
    deinterleave is ``tl.split``, not ``reshape`` + ``permute``.
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
        # it is the (SW=2, WQ) reorder.
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

    _pad_split_cast_kernel with the halo materialisation folded away: the padding shift
    goes into the split-store and every border is left to the memset that precedes it.
    The only mask is the block/width overshoot, written as a *value* mask (other=0),
    which tl.split carries correctly unlike a clamped index.
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

    _fused_split_cast_kernel reads a whole input row, which overflows the unified
    buffer for wide 1-D rows, so cut the same fold along the width.  H == 1 and PH == 0,
    so (n, c, h) is one flat axis of N*C rows, and the launcher requires
    W % BLOCK_W == 0 and N*C % BLOCK_ROWS == 0 to keep the load mask-free.

    The pad is folded into the store as in _fused_split_cast_kernel, with the column
    origin at ``c0 = pid_w * BLOCK_W``.
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

    The zero-then-overwrite shape of _fused_split_cast at H == 1, PH == 0: the row axis
    is N*C, the input is a flat (N*C, W) run, and the padding q-columns the split-store
    leaves behind are zeroed by the preceding memset.  The width is tiled by BLOCK_W so
    a wide row does not overflow the unified buffer.

    ``weight`` folds the _prep_weight permutation onto the memset fill, so the call
    returns ``(out, wt)`` and the prep's own launch is skipped.
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
    # Full-buffer memset, as in _fused_split_cast.  A strip-only zero is a strided
    # single-column store, which this backend issues one lane at a time, so zeroing
    # the whole buffer contiguously is the cheaper pass.
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

    Zero-fill the whole split buffer with one flat memset, then
    _fused_split_cast_kernel overwrites the interior straight from ``input``.
    SW=2 and SW=4 with any PW < 2*SW are handled by the same plane-rotation rule.
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
    # BLOCK=4096 holds the memset at its bandwidth ceiling; the fill is the one
    # pass this path adds over the vendor pair.
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

    ``_pad_input`` builds the halo, then ``_pad_split_cast_kernel`` reads it mask-free
    and deinterleaves with ``tl.split`` while upcasting to ``arith``.  Bit-identical to
    ``_pad_split_width(input.to(arith), ...)``; SW=2 and SW=4 are the only two strides
    it takes and any other falls back.

    Returns ``(src, wt)``; ``wt`` is None unless the 1-D fused path fired with a
    ``weight``.
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

    # The 4-way split reads SW=4 times as many columns per row, so its register
    # tile is 2x the 2-way one and must shrink to stay inside the unified buffer.
    block_rows = _SPLIT_BLOCK_ROWS // 2 if sw == 4 else _SPLIT_BLOCK_ROWS
    block_q = _SPLIT_BLOCK_Q // 2 if sw == 4 else _SPLIT_BLOCK_Q

    # Fused path: build the split halo straight from the unpadded input, folding
    # the pad into the split-store.  Needs a contiguous input and a whole-row load
    # that fits the unified buffer.
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
    # buffer for a wide row, so tile the width instead.  Needs W a power of two and
    # N*C % BLOCK_ROWS == 0 so the load stays mask-free.  Stride 4 is held back
    # because its only case is already well covered and not bottlenecked here.
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
    # buffer's masked-lane allowance (see _slack).
    ow_max = triton.cdiv(out_w, block_w) * block_w - 1 + (kw - 1) * dw // sw
    tail = ow_max + 1 - wq
    numel = rows * sw * wq
    buf = torch.empty(numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=arith)
    out = buf[:numel].view(N, C, Hp, sw, wq)
    out_s = out.stride()

    # _pad_split_cast_kernel's load must stay mask-free -- a masked or clamped
    # index folded through reshape + tl.split silently zeroes the tile on this
    # backend -- so rows must divide BLOCK_ROWS or the shape falls back to
    # _pad_split_width.
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

    A (BLOCK_C, BLOCK_W) input tile and a (BLOCK_OC, BLOCK_C) weight tile per tap are
    reused for BLOCK_OC * BLOCK_W outputs, so the load is ``1/BLOCK_W + 1/BLOCK_OC``
    bytes per MAC.  BLOCK_W wins the tie under _BLOCK_ELEMS because it is the run
    length: take the whole row, give the channel tile the remainder.
    """
    cap_oc = min(_BLOCK_OC_MAX, max(1, triton.next_power_of_2(oc_per_group)))
    block_w = min(block_w_cap or _BLOCK_W_MAX, max(1, triton.next_power_of_2(ow)))
    block_oc = min(cap_oc, max(1, _BLOCK_ELEMS // block_w))
    return block_oc, block_w


def _can_use_dot(block_oc, block_w, c_in):
    """Whether the whole tile satisfies tl.dot's minimum dimension.

    A shape that fails falls back to the FMA kernel rather than padding up to 16: the
    kernel *can* pad, but in bf16/fp16 the 16/3 arithmetic costs more than the FMA
    chain it replaces.  Only fp32 comes out ahead, and _direct_conv2d exploits that by
    forcing ``use_dot`` on a dot-sized output tile and pinning ``arith`` to fp32.

    The channel count is asked for *rounded up*, since that is the tile the kernel
    builds; rounding up may only *double* the tile (c_in of 8 and above).
    """
    tile_c = triton.next_power_of_2(c_in)
    if c_in >= _DOT_MIN // 2:
        tile_c = max(tile_c, _DOT_MIN)
    return min(block_oc, block_w, tile_c) >= _DOT_MIN


def _pick_block_c(c_in, use_dot, block_w):
    """Channel tile for the reduction, in elements.

    Only meaningful on the dot path.  Capped at _BLOCK_C_MAX so a very wide channel
    dim is reduced in a few dots, and rounded up so a count between powers of two still
    gets one dot with a masked tail.  Then shrunk until the input tile fits the unified
    buffer -- give up channels, not BLOCK_W.
    """
    if not use_dot:
        return 1
    # Floored at _DOT_MIN, matching _can_use_dot: a BLOCK_C of 8 would not be a dot
    # at all.
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
    arange/IndexPutV2 pair: the element is either on the diagonal, loaded from
    ``w[oc, 0, t]``, or a zero this store writes directly.  The source address is
    in bounds for every lane, masked or not, so this needs no slack on the input.
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

    The tap and channel axes are flattened into one ``ct = c * T + t`` so the tile is a
    plain 2-D block whose fast axis is the *source's* contiguous one -- which is what
    lets the MTE issue a run per row instead of a gather.  The transpose into the
    destination's ``oc`` fast axis is unavoidable; the load side does not have to
    scatter as well, and that is the whole win over the vendor copy.

    ``C``, ``OC`` and ``T`` are constexpr so ``ct // T`` is a compile-time multiply
    rather than the per-lane emulated division a runtime divisor falls back to.
    """
    ct = tl.program_id(0) * BLOCK_CT + tl.arange(0, BLOCK_CT)
    oc = tl.program_id(1) * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ct_ok = ct < C * T
    oc_ok = oc < OC
    c = ct // T
    t = ct - c * T
    # Addressed with ``ct`` and not with ``c * T + t``.  The two are the same
    # number and are not the same address expression: ``c`` and ``t`` come out of a
    # division, so the sum has no provable stride and the backend issues it as a
    # gather.  As ``ct`` the row is one contiguous run and the MTE issues it as one.
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

    Depthwise is a dense convolution whose weight happens to be block diagonal with 1x1
    blocks, so writing the blocks out and running the ordinary dense kernel computes
    the same sum -- every product is exact, only the accumulation order changes.  It
    costs ``groups`` times the arithmetic, which beats collapsing BLOCK_OC to 1 and
    leaving the FMA kernel one rank-one update per program.  Only the 1x1-block case is
    worth it; a wider group clears tl.dot's minimum anyway.
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

    The largest tile each axis allows is not the fastest: it minimises programs, and
    these tensors are not bandwidth bound.  A first-axis extent of 8 or 16 with the
    second axis as wide as it was is at or within noise of the best of ~30 (ct, oc)
    pairs per shape.

    The second axis is *not* free to shrink: it is the kernel's store row.
    """
    block_oc = min(_PREP_BLOCK, triton.next_power_of_2(oc))
    if ct_n <= _PREP_CT_MAX_ELTS:
        block_ct = _PREP_CT_SMALL
    else:
        block_ct = _PREP_CT_LARGE
    return min(block_ct, triton.next_power_of_2(ct_n)), block_oc


def _prep_weight(weight, out_dtype=None):
    """Move the output-channel axis last and make it contiguous.

    The kernels read one weight vector per (tap, channel) pair, indexed by output
    channel.  In the native (OC, C, KH, KW) layout that vector has stride C*KH*KW -- a
    gather in the innermost loop.  (KH, KW, C, OC) makes it unit-stride, at the cost of
    one small transpose per call.

    The result lands in a buffer with _SLACK_ELEMS to spare: the kernel's masked
    output-channel lanes still form an address past the end of it (see _slack).
    _prep_weight_kernel does the transpose in one launch and takes ``out_dtype``, so the
    FMA arm's cast rides along instead of costing a second kernel.
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
    suppress the *access*, not the address, and the MTE faults (507015) before the mask
    is consulted.  This slack is what makes _pad_flat free to leave its rows unwidened;
    rounding up to _SLACK_ELEMS also covers the allocator's alignment, which is what
    makes the fault look intermittent.
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
    # Tap-major loop order: the tap offset and the tap pointer depend only on (kh, kw),
    # so hoisting them out of the channel loop is one address computation per tap
    # instead of one per tap and channel.
    #
    # The tap indices are the *unpadded* ones, so the launcher's zero halo is what makes
    # them valid and the loads carry no mask; a clamp instead takes the address out of
    # the affine form the backend needs, for an order of magnitude.
    #
    # Both arms share this nest: a constexpr `if` prunes its dead arm, so neither pays
    # for the other and the addressing cannot drift apart in two copies.
    if RUNTIME_TAPS:
        # Large kernels walk their taps in a *runtime* loop: unrolled bf16 dots hang the
        # device (see _arith_dtype), so any tap count past _DOT_TAPS_MAX needs a loop, and a
        # runtime loop holds one dot in the body however many taps it makes.  The fp32 form
        # also beats the unrolled one by staggering the tap loads instead of bursting them.
        # RUNTIME_TAPS is only ever set with USE_DOT, so the FMA arm is absent here.
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
                    # The launcher reshaped the halo's width axis to (W/SW, SW) planes, so tap kw
                    # lives in plane (kw*DW) % SW at column ow + (kw*DW)//SW -- unit stride in ow,
                    # with the plane a launch-time constant.  See _pad_split_width.
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
                    # One (BLOCK_OC, BLOCK_C) x (BLOCK_C, BLOCK_W) dot per tap and channel block.
                    # This is what puts the operator on the cube; the same work as FMA runs on the
                    # vector unit instead.
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

    The tap address of output position (oh, ow) is

        (oh + kh) * Wp + (ow + kw)   =   p + kh * Wp + kw,   p = oh * Wp + ow

    a *constant* shift, provided the padded input's row stride and the output plane's
    row stride are both Wp.  A program can then own a flat interval of p spanning many
    rows, and the x load becomes (BLOCK_C, BLOCK_M) with BLOCK_M contiguous bytes per
    row instead of BLOCK_W -- a run length the blocked kernel cannot reach, capped by
    the tap count, which is paid in MTE2 request slots rather than absorbed by cache.

    The price is the output layout: the result is written Wp-strided and compacted back
    to OW columns by _compact_plane_kernel.  Wp must be the *unwidened* padded width,
    W + 2*PW: the only overshoot is the last program's, so the room is a single tail on
    the allocation rather than columns on every row.
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
                # The channel mask sits on the *weight*, not on x, and only on the last channel
                # block: an x lane that overshoots C_IN reads the next channel, which is
                # garbage but in bounds given the tail, and the zeroed weight discards it.  A
                # masked load would put a compare on the tile that matters.
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

    # ``oc_mask`` is in the store mask for the reason the channel overshoot above is
    # harmless: a lane past OC keeps its address and the output plane's channel
    # stride walks it into the next image.  Only the 3-D launcher can raise BLOCK_OC
    # past what the shape fills, but the mask costs nothing to add.
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

    Small and unavoidable, and it costs far more than the bytes it moves: the cost is
    the row stride, not the mask, and every tiling that avoids the stride is worse -- a
    row per program, or a per-lane column rebuild.

    Do not fuse the compaction into the kernel that produces the plane: ``oh * OW + ow``
    is not affine in ``p``, so the store's stride becomes unprovable and emits a gather.
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
    # Channel-major, unlike the 2D kernel: a runtime channel loop inside a
    # 27-times-unrolled tap loop rather than around it sends the Ascend compiler into a
    # multi-minute compile -- three times the per-tap code with no extra parallelism.
    # Nine taps is under that limit; twenty-seven is not.
    #
    # Both arms share this nest (a constexpr `if` prunes its dead arm), and the taps stay
    # unpadded so the loads stay unmasked, as in 2D.
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

    The 2-D kernel's identity, one dimension at a time.  A tap of output position
    (od, oh, ow) reads the padded input at

        (od + kd*DD)*Dp*Hp*Wp + (oh + kh*DH)*Wp + (ow + kw*DW)
      = od*Dp*Hp*Wp + [(oh*Wp + ow) + kd*DD*Dp*Hp*Wp + kh*DH*Wp + kw*DW]

    so with the output's flat index taken over one depth slice every tap is again a
    constant shift of ``p``, and the depth axis rides on the grid as ``od`` for free.

    Wp is the un-widened padded width, as in 2-D, and the price is the same: the result
    is written Wp-strided per depth slice and compacted back to OW columns by
    _compact_plane_kernel.
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
                    # The channel mask sits on the weight, not on x, for the reason given in the
                    # 2-D kernel.
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
# ``tl.arange`` and no loop, so a wider padded row falls back to the strided
# copy there.
_PAD_FLAT_W_MAX = 128


def _pad_input_flat(input, padding, tail):
    """``_pad_input``'s halo with the spare room as one tail on the allocation.

    ``_pad_input`` widens *every* padded row by ``tail``; the flat kernel has no
    per-tile masked lanes -- only the very last program overshoots -- so the rows stay
    exactly ``W + 2*PW`` wide and the room is one tail past the end.  The rows have to
    be un-widened for the flat identity to hold: the pad's row stride *is* the output
    plane's row stride.

    The 5-D interior uses the same affine triton pair, not the strided ``copy_``, which
    for a 5-D destination lands on aclnn's host-side AiCpu ViewCopy.
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
        # No halo to write, only room to leave, so one contiguous pass and the buffer
        # stays uninitialised: with no padding there is nothing for a zero to stand in
        # for, and the masked-lane loads past the end never store.
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
# The flat kernel's accumulator is (BLOCK_OC, BLOCK_M) in fp32 and it is what the
# unified buffer is spent on.  _BLOCK_ELEMS is the same 32768 the blocked
# launcher already tiles by, reached from the other side.
_FLAT_ACC_MAX = 32768


def _pick_flat_blocks(oc_per_group, tot_p, floor_oc=1):
    """Tile for the flat kernels: output channels first, run length second.

    BLOCK_OC sets how many output channels share one x tile, so it divides the x
    traffic; BLOCK_M only sets the length of each of those runs.  Spend on BLOCK_OC,
    then give the remainder to BLOCK_M.

    ``floor_oc`` raises the channel tile past what the shape fills, which only the 3-D
    launcher asks for: a count under _DOT_MIN would otherwise cost the kernel its dot.
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
# Ceiling on the compaction's traffic, as a multiple of the x traffic the conv reads
# for the output being compacted: the compaction is a read *and* a write of the
# output plane against KH*KW passes of the padded input.  Widen it only for a shape
# whose run genuinely gets longer -- a high ratio with a narrow strided x read does
# not justify it.
_FLAT_COMPACT_MAX = 0.5


def _use_flat_2d(input, weight, padding, groups, oh, ow, use_dot):
    """Is the flat kernel both legal and worth its compaction on this shape?

    Legality: stride 1, because the constant tap offset is the flat image of
    ``oh*SH + kh*DH == (oh+kh)*DH``; the dot, because it is the kernel's only arm; and
    the native dtype, because that is the regime every tile bound was measured in.

    Worth: the run has to actually get longer (see _pick_flat_blocks), and the
    compaction has to stay small against the x traffic it saves.
    """
    if not use_dot or input.dtype not in (torch.float16, torch.bfloat16):
        return False
    N, C, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    if KH * KW > _DOT_TAPS_MAX or OC // groups < _DOT_MIN:
        return False
    # A 1-D shape lifted to H==1 never wins on flat: its blocked run is already the
    # whole row, so the flat identity only adds the compaction.
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
        # A C_in below _DOT_MIN is forced onto the padded-dot path, whose x read is a
        # strided gather far below the byte-rate the ratio above assumes; admit a higher
        # ratio there.
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
    # The channel tile is the only thing that can overshoot C_IN, and only on its
    # last block: the weight is masked to zero there, so the garbage x lanes are
    # harmless -- but their addresses still have to exist.
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

    Same two questions as _use_flat_2d and the same answers: stride 1 in all three axes,
    a run that actually gets longer, and a compaction small against the x traffic it
    saves.

    The dtype clause the 2-D gate carries is not here: the 3-D launcher stays in fp32.
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

    fp32 unconditionally, where the 2-D launcher lets the native dtype through on the
    dot path: these tiles sit at tl.dot's minimum, where the cube is latency-bound and
    bf16 measured *slower*.  Flat changes the tile, but that measurement has not been
    redone, so this stays as the blocked path had it.

    The output is one ``(OD, OH, Wp)`` slab per channel, compacted by
    _compact_plane_kernel.
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
    # Not _pick_block_c: its _UB_TILE_MAX shrink is the *blocked* kernel's budget.
    # Here the accumulator is BLOCK_OC * BLOCK_M and the x tile rides alongside it,
    # so the only floor that matters is tl.dot's.
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

    # _can_use_dot refuses a contraction axis below _DOT_MIN//2 (c_in < 8) because
    # padding it to 16 triples the dot's arithmetic, which costs in bf16/fp16 but
    # wins in fp32.  Force the dot path when the output tile is already dot-sized and
    # let the fp32 override below pay for the padded contraction.
    if not use_dot and block_oc >= _DOT_MIN and block_w >= _DOT_MIN:
        use_dot = True

    # Depthwise is the one grouped shape the per-group tile cannot carry: with one
    # input and one output channel per group, BLOCK_OC is 1 and the tile never
    # reaches tl.dot.  Densifying puts the same arithmetic on the cube.
    if not use_dot and groups > 1 and OC // groups == 1 and weight_c == 1:
        dense = _densify_depthwise(weight, input.shape[1])
        dense_oc, _ = _pick_blocks(OW, OC)
        if _can_use_dot(dense_oc, block_w, dense.shape[1]):
            return _direct_conv2d(input, dense, padding, stride, dilation, 1)

    block_c = _pick_block_c(weight_c, use_dot, block_w)
    # Large kernels walk their taps in a runtime loop (one dot in the unrolled body)
    # so the code stays bf16-capable without the unrolled-bf16 hang; see
    # _arith_dtype.  Whether the loop keeps bf16 is decided after split_w below.
    runtime_taps = use_dot and KH * KW > _DOT_TAPS_MAX
    split_w = SW >= _SPLIT_MIN_STRIDE
    arith = _arith_dtype(
        input, use_dot, KH * KW, runtime_taps=runtime_taps and not split_w
    )
    # A dot forced onto a sub-16 contraction pads its channel tile to _DOT_MIN, and
    # that padding is only worth it in fp32; _arith_dtype would otherwise keep bf16
    # here, so pin it down.
    if use_dot and weight_c < _DOT_MIN:
        arith = torch.float32

    # Stride-1 shapes index the plane flat, so the x tile's innermost run is BLOCK_M
    # instead of BLOCK_W -- a length the blocked form cannot reach.  It costs a
    # compaction of the Wp-strided result back to OW columns.
    if (
        SH == 1
        and SW == 1
        and _use_flat_2d(input, weight, padding, groups, OH, OW, use_dot)
    ):
        return _flat_conv2d(input, weight, padding, dilation, groups, arith)

    # The halo is only built when the padding is non-zero, the only thing that can put a
    # tap outside the input; a row that is not a whole number of tiles carries ``w_ok``
    # on the load instead.
    #
    # The upcast happens here rather than inside the pad (``arith`` is fp32 except on the
    # dot path; see _arith_dtype).  A width stride turns every tap load into a stride-SW
    # run; the split halo turns it back into a contiguous one and needs the copy whether
    # or not there is padding to write.
    #
    # A 1x1 kernel has nothing to permute: at T == 1 the source's own (OC, C) already has
    # the unit-stride axis the kernel walks per tap, so the tile is one contiguous run
    # read straight from the caller's tensor and a launch is saved.  Gated on the dot arm
    # because the FMA arm reads its weight as a vector over oc; the masks and tile
    # divisibility are required, not preferred -- a masked lane would compute an address
    # past the end of the *caller's* tensor, which has none of _prep_weight's slack.
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
    # The weight permutation rides along with the 1-D halo when there is one to ride along
    # with, and is built on its own otherwise; both entries land in `wt`.
    #
    # The upcast is not a launch of its own when the halo kernel is going to write every
    # element anyway -- it costs one there, a wider store.
    #
    # Only the contiguous case folds: the kernel reads its source as ``r*W + c``, and
    # ``cudnn_convolution`` materialises a strided operand before dispatching here.
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

    # Widening BLOCK_W past a narrow OW to reach the dot arm was tried and is a loss:
    # these tiles sit at tl.dot's minimum, where 27 dots of (16,16)x(16,16) are
    # latency-bound, while the FMA arm's rank-one updates at least keep the vector
    # unit issuing.

    # Same depthwise lift as the 2D launcher and for the same reason; see
    # _densify_depthwise.  The 3D variant was missing it, which pins BLOCK_OC to 1
    # and leaves the FMA kernel doing 27 rank-1 updates per output.
    if not dot_ok and groups > 1 and OC // groups == 1 and weight_c == 1:
        dense = _densify_depthwise(weight, input.shape[1])
        dense_oc, _ = _pick_blocks(OW, OC)
        if _can_use_dot(dense_oc, block_w, dense.shape[1]):
            return _direct_conv3d(input, dense, padding, stride, dilation, 1)

    use_dot = dot_ok and _DOT_3D
    block_c = _pick_block_c(weight_c, use_dot, block_w)

    # Stride-1 shapes index the (oh, ow) plane flat, one depth slice per program; see
    # _flat_conv3d_kernel.  Taken before the halo because it builds a different one --
    # the un-widened rows the flat identity needs.
    if _use_flat_3d(input, weight, padding, stride, groups, OD, OH, OW):
        return _flat_conv3d(input, weight, padding, dilation, groups)

    # Same halo rule as the 2D launcher; see the note there.  The dtype is fp32
    # unconditionally, not gated on use_dot as the 2D launcher's is: the 3-D dot arm
    # walks its taps in a runtime loop, and handing it native bf16 tiles measured
    # 2.3-3.0x *slower* because these tiles sit at tl.dot's minimum and the cube is
    # latency-bound there.
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
    # One (n, oh, ow-chunk) row per program over the M axis; the output-channel block
    # is the N axis.  n and oh are scalars so the halo's padded-coordinate index stays
    # a scalar+offset rather than a gather; ow is the contiguous axis.
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
    # k = tap*C + c -- the big-K contraction the cube is built for.
    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_ok = k_off < K
        c = k_off % C
        # Clamp the tail tap so a masked lane's address stays on the last valid tap (see
        # _slack); the mask zeroes the lane but the address is still formed.
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
    K = KH*KW*C_in, N = OC.  The im2col is implicit -- the GEMM's A-load gathers the
    window directly -- so the contraction axis is the full KH*KW*C_in instead of the
    direct kernel's BLOCK_C, at the price of a real gather.
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

    CANN's default is im2col+GEMM; it only steps aside for 1x1, dilated kernels,
    depthwise/grouped, 3-D, and a contraction under tl.dot's minimum.  Stride does not
    disqualify -- the direct kernel pays a split-width transpose for stride >= 2 that
    im2col folds into the gather for free.
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

    Dimensions, parameter normalization and the returned layout follow the shared
    implementation in ``flag_gems/ops/cudnn_convolution.py``; only the arithmetic and
    the tiling differ.  ``benchmark``, ``deterministic`` and ``allow_tf32`` are accepted
    for interface compatibility and ignored.
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

    # The halo builders index the operand as if it were contiguous: _pad_input's interior
    # copy at ``plane * H * W + r * W + c``, _pad_flat_row's 1-D lift at ``r * W + c``.
    # So a strided operand must be materialised before either is reached, whatever its
    # dtype -- a strided fp32 operand is the case that bit, since ``arith == input.dtype``
    # makes ``.to(arith)`` the identity.  The conv kernels themselves take a stride list
    # and are fine; this is about the pad alone.
    if not input.is_contiguous():
        input = input.contiguous()

    # Depthwise used to be handed to the shared kernel on the grounds that it has no
    # cross-channel reduction.  That is true of the *arithmetic* and false of the
    # memory access: the shared tile addresses its input through a gather per tap,
    # which is the one thing this backend charges for.  Depthwise is the shape the
    # direct path suits best -- groups == C_in collapses BLOCK_OC to 1, so there is
    # no masked-out outer product to pay for either way.
    if input.dtype == torch.float32:
        return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)

    # --- fp16 / bf16 -------------------------------------------------------
    # Everything else goes to the direct kernels, in the dtype each asks for; only the
    # tl.dot branch keeps the native dtype (see _arith_dtype).  A 1x1 kernel with one
    # group, unit stride and no padding is a plain GEMM and used to go to the shared
    # pointwise kernel: the direct path runs it as one dot over the whole channel
    # reduction and is faster in every case tried.
    #
    # Handing bf16 to a kernel that is *not* on the dot path is a large loss -- the
    # backend does not vectorize an in-loop bf16 load feeding a multiply -- so those
    # paths convert once up front.  It also fixes a correctness hole: the shared
    # conv1d/2d/3d kernels mis-compute 1x1 kernels in every dtype against an fp64
    # reference.
    #
    # There is deliberately no channel bound here: the one that used to exist was
    # measured against a kernel that clamped every tap to stay in bounds, and the
    # materialised halo removed exactly the per-tap cost it was reasoning about.
    return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)
