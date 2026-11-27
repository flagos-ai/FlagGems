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

# Kunlunxin (XPU) specialization of aten::grid_sampler_2d_backward.
#
# The generic implementation (src/flag_gems/ops/grid_sampler_2d_backward.py)
# is silently wrong on this backend for two reasons, both shared with the 3d
# sibling (see grid_sampler_3d_backward.py in this directory):
#
# 1. `tl.atomic_add(..., mask=<data-dependent composite mask>)` does NOT honour
#    the mask, so out-of-bounds bilinear corners are accumulated at their
#    clamped (in-bounds) addresses.  The generic kernel also issues
#    `tl.load(addr, mask=False, other=0.0)` for out-of-bounds corners; `other=`
#    is not honoured here, so those loads return garbage instead of zero and
#    poison grad_grid.  Fix: clamp every corner address to a legal location and
#    gate the *payload* with `tl.where`, keeping only the tail mask on the
#    atomic/load.
#
# 2. `tl.atomic_add` loses updates when several *programs* touch the same
#    address concurrently.  The generic kernel launches a grid of (N, OH, C),
#    so the OH programs of one (n, c) race on grad_input[n, c] and the C
#    programs of one (n, oh) race on grad_grid[n, oh].  Fix: give every program
#    exclusive ownership of one grad_input[n, c] slice (grid = (N*C,), runtime
#    loop over output-point tiles), and compute grad_grid with a separate
#    kernel that owns one output point per program and reduces over C with
#    tl.sum.
#
# The reflection pad multiplier (gradient of the folded+clipped coordinate) is
# the continuous reflection derivative: the triangle-wave fold slope (+/-1)
# times an indicator that the pre-clip reflected coordinate lands strictly
# inside (0, size-1).  This matches the CPU reference at every fold/clip
# boundary (the multi-period `_reflect` below folds an arbitrary number of
# periods, unlike the single-fold helper in the 3d file).

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Verified-safe channel tile widths on this backend are 64 and 128; 4 and 16
# both produce `CUDA error: unspecified launch failure`.  Wider widths are not
# validated, so large C is handled by looping the launch on the host.
_MAX_BLOCK_C = 128
_MIN_BLOCK_C = 64


@triton.jit
def _src_index(coord, size, align_corners: tl.constexpr):
    if align_corners:
        idx = (coord + 1.0) * 0.5 * (size - 1)
    else:
        idx = ((coord + 1.0) * size - 1.0) * 0.5
    return idx


@triton.jit
def _clip(coord, size):
    return tl.minimum(tl.maximum(coord, 0.0), size - 1.0)


@triton.jit
def _reflect(coord, size, align_corners: tl.constexpr):
    # Multi-period continuous reflection, identical in value to the generic
    # `_reflect_coord` (without its final [0, size-1] clip, which is applied
    # separately by `_clip` in `_pad`).
    if align_corners:
        mn = 0.0
        span = size - 1.0
    else:
        mn = -0.5
        span = size * 1.0
    in_abs = tl.abs(coord - mn)
    extra = in_abs % span
    flips = tl.floor(in_abs / span)
    ref = tl.where(flips % 2 == 0, extra, span - extra) + mn
    return tl.where(span == 0.0, 0.0, ref)


@triton.jit
def _reflect_grad(coord, size, align_corners: tl.constexpr):
    # d(_clip(_reflect(coord)))/d(coord), the continuous reflection derivative.
    # Fold slope is (+1) on even periods and (-1) on odd periods, times the sign
    # of (coord - min); the outer clip contributes 1 strictly inside (0, size-1)
    # and 0 in the saturated bands / exactly on the boundaries.
    if align_corners:
        mn = 0.0
        span = size - 1.0
        u = coord
    else:
        mn = -0.5
        span = size * 1.0
        u = coord + 0.5
    ax = tl.abs(coord - mn)
    extra = ax % span
    flips = tl.floor(ax / span)
    fold = tl.where(flips % 2 == 0, extra, span - extra) + mn
    sign = tl.where(u >= 0.0, 1.0, -1.0)
    reflect_grad = tl.where(flips % 2 == 0, 1.0, -1.0) * sign
    clip_grad = tl.where((fold > 0.0) & (fold < size - 1.0), 1.0, 0.0)
    return tl.where(span == 0.0, 0.0, reflect_grad * clip_grad)


@triton.jit
def _clip_grad(coord, size):
    # d(_clip(coord))/d(coord): 1 strictly inside (0, size-1), 0 elsewhere
    # (border padding saturates exactly on the boundaries).
    return tl.where((coord > 0.0) & (coord < size - 1.0), 1.0, 0.0)


@triton.jit
def _pad(coord, size, padding_mode: tl.constexpr, align_corners: tl.constexpr):
    if padding_mode == 1:
        c = _clip(coord, size)
    elif padding_mode == 2:
        c = _clip(_reflect(coord, size, align_corners), size)
    else:
        c = coord
    return c


@triton.jit
def _grad_input_bilinear_kernel(
    go_ptr,
    grid_ptr,
    gi_ptr,
    C,
    iH,
    iW,
    oH,
    oW,
    go_sN,
    go_sC,
    gi_sN,
    gi_sC,
    grid_sN,
    grid_sH,
    grid_sW,
    n_per_batch,
    n_tiles,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    go_base = n * go_sN + c * go_sC
    gi_base = n * gi_sN + c * gi_sC
    isH = iW
    for t in range(n_tiles):
        p = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = p < n_per_batch
        ow = p % oW
        oh = p // oW
        gb = n * grid_sN + oh * grid_sH + ow * grid_sW
        gx = tl.where(pm, tl.load(grid_ptr + gb, mask=pm), 0.0)
        gy = tl.where(pm, tl.load(grid_ptr + gb + 1, mask=pm), 0.0)
        ix = _pad(_src_index(gx, iW, align_corners), iW, padding_mode, align_corners)
        iy = _pad(_src_index(gy, iH, align_corners), iH, padding_mode, align_corners)
        ix0 = tl.floor(ix).to(tl.int32)
        iy0 = tl.floor(iy).to(tl.int32)
        tx = ix - ix0.to(tl.float32)
        ty = iy - iy0.to(tl.float32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        x0ok = (ix0 >= 0) & (ix0 < iW)
        x1ok = (ix1 >= 0) & (ix1 < iW)
        y0ok = (iy0 >= 0) & (iy0 < iH)
        y1ok = (iy1 >= 0) & (iy1 < iH)
        x0s = tl.maximum(tl.minimum(ix0, iW - 1), 0)
        x1s = tl.maximum(tl.minimum(ix1, iW - 1), 0)
        y0s = tl.maximum(tl.minimum(iy0, iH - 1), 0)
        y1s = tl.maximum(tl.minimum(iy1, iH - 1), 0)
        g = tl.where(pm, tl.load(go_ptr + go_base + p, mask=pm).to(tl.float32), 0.0)
        zero = tl.zeros([BLOCK_P], dtype=tl.float32)
        a0 = y0s * isH
        a1 = y1s * isH
        tl.atomic_add(
            gi_ptr + gi_base + a0 + x0s,
            tl.where(x0ok & y0ok, g * (1.0 - tx) * (1.0 - ty), zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + gi_base + a1 + x0s,
            tl.where(x0ok & y1ok, g * (1.0 - tx) * ty, zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + gi_base + a0 + x1s,
            tl.where(x1ok & y0ok, g * tx * (1.0 - ty), zero),
            mask=pm,
        )
        tl.atomic_add(
            gi_ptr + gi_base + a1 + x1s,
            tl.where(x1ok & y1ok, g * tx * ty, zero),
            mask=pm,
        )


@triton.jit
def _grad_input_nearest_kernel(
    go_ptr,
    grid_ptr,
    gi_ptr,
    C,
    iH,
    iW,
    oH,
    oW,
    go_sN,
    go_sC,
    gi_sN,
    gi_sC,
    grid_sN,
    grid_sH,
    grid_sW,
    n_per_batch,
    n_tiles,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    go_base = n * go_sN + c * go_sC
    gi_base = n * gi_sN + c * gi_sC
    isH = iW
    for t in range(n_tiles):
        p = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = p < n_per_batch
        ow = p % oW
        oh = p // oW
        gb = n * grid_sN + oh * grid_sH + ow * grid_sW
        gx = tl.where(pm, tl.load(grid_ptr + gb, mask=pm), 0.0)
        gy = tl.where(pm, tl.load(grid_ptr + gb + 1, mask=pm), 0.0)
        ix = _pad(_src_index(gx, iW, align_corners), iW, padding_mode, align_corners)
        iy = _pad(_src_index(gy, iH, align_corners), iH, padding_mode, align_corners)
        # Round to nearest, half to even (matching the generic/forward kernels).
        x_floor = tl.floor(ix)
        y_floor = tl.floor(iy)
        x_frac = ix - x_floor
        y_frac = iy - y_floor
        x_floor_int = x_floor.to(tl.int32)
        y_floor_int = y_floor.to(tl.int32)
        x_up = (x_frac > 0.5) | ((x_frac == 0.5) & ((x_floor_int & 1) == 1))
        y_up = (y_frac > 0.5) | ((y_frac == 0.5) & ((y_floor_int & 1) == 1))
        xn = x_floor_int + x_up.to(tl.int32)
        yn = y_floor_int + y_up.to(tl.int32)
        ok = (xn >= 0) & (xn < iW) & (yn >= 0) & (yn < iH)
        xs = tl.maximum(tl.minimum(xn, iW - 1), 0)
        ys = tl.maximum(tl.minimum(yn, iH - 1), 0)
        g = tl.where(pm, tl.load(go_ptr + go_base + p, mask=pm).to(tl.float32), 0.0)
        zero = tl.zeros([BLOCK_P], dtype=tl.float32)
        tl.atomic_add(
            gi_ptr + gi_base + ys * isH + xs,
            tl.where(ok, g, zero),
            mask=pm,
        )


@triton.jit
def _grad_grid_bilinear_kernel(
    go_ptr,
    inp_ptr,
    grid_ptr,
    gg_ptr,
    C,
    c_begin,
    c_end,
    iH,
    iW,
    oH,
    oW,
    go_sN,
    go_sC,
    inp_sN,
    inp_sC,
    grid_sN,
    grid_sH,
    grid_sW,
    gg_sN,
    gg_sH,
    gg_sW,
    padding_mode: tl.constexpr,
    align_corners: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % oW
    t1 = pid // oW
    oh = t1 % oH
    n = t1 // oH
    gb = n * grid_sN + oh * grid_sH + ow * grid_sW
    gx = tl.load(grid_ptr + gb)
    gy = tl.load(grid_ptr + gb + 1)

    rx = _src_index(gx, iW, align_corners)
    ry = _src_index(gy, iH, align_corners)

    if align_corners:
        mx = 0.5 * (iW - 1)
        my = 0.5 * (iH - 1)
    else:
        mx = 0.5 * iW
        my = 0.5 * iH
    if padding_mode == 1:
        px = _clip_grad(rx, iW)
        py = _clip_grad(ry, iH)
    elif padding_mode == 2:
        px = _reflect_grad(rx, iW, align_corners)
        py = _reflect_grad(ry, iH, align_corners)
    else:
        px = 1.0
        py = 1.0

    ix = _pad(rx, iW, padding_mode, align_corners)
    iy = _pad(ry, iH, padding_mode, align_corners)

    ix0 = tl.floor(ix).to(tl.int32)
    iy0 = tl.floor(iy).to(tl.int32)
    tx = ix - ix0.to(tl.float32)
    ty = iy - iy0.to(tl.float32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    x0ok = (ix0 >= 0) & (ix0 < iW)
    x1ok = (ix1 >= 0) & (ix1 < iW)
    y0ok = (iy0 >= 0) & (iy0 < iH)
    y1ok = (iy1 >= 0) & (iy1 < iH)
    x0s = tl.maximum(tl.minimum(ix0, iW - 1), 0)
    x1s = tl.maximum(tl.minimum(ix1, iW - 1), 0)
    y0s = tl.maximum(tl.minimum(iy0, iH - 1), 0)
    y1s = tl.maximum(tl.minimum(iy1, iH - 1), 0)
    isH = iW
    q00 = y0s * isH + x0s
    q01 = y0s * isH + x1s
    q10 = y1s * isH + x0s
    q11 = y1s * isH + x1s

    cs = c_begin + tl.arange(0, BLOCK_C)
    cm = cs < c_end
    csf = tl.where(cm, cs, 0)
    g = tl.load(go_ptr + n * go_sN + csf * go_sC + oh * oW + ow).to(tl.float32)
    g = tl.where(cm, g, 0.0)
    ib = n * inp_sN + csf * inp_sC
    i00 = tl.load(inp_ptr + ib + q00).to(tl.float32)
    i01 = tl.load(inp_ptr + ib + q01).to(tl.float32)
    i10 = tl.load(inp_ptr + ib + q10).to(tl.float32)
    i11 = tl.load(inp_ptr + ib + q11).to(tl.float32)
    zero = tl.zeros([BLOCK_C], dtype=tl.float32)
    i00 = tl.where(x0ok & y0ok, i00, zero)
    i01 = tl.where(x1ok & y0ok, i01, zero)
    i10 = tl.where(x0ok & y1ok, i10, zero)
    i11 = tl.where(x1ok & y1ok, i11, zero)
    ax = g * ((1.0 - ty) * (i01 - i00) + ty * (i11 - i10))
    ay = g * ((1.0 - tx) * (i10 - i00) + tx * (i11 - i01))
    sx = tl.sum(ax) * mx * px
    sy = tl.sum(ay) * my * py
    ggb = n * gg_sN + oh * gg_sH + ow * gg_sW
    if ACCUMULATE:
        sx = sx + tl.load(gg_ptr + ggb)
        sy = sy + tl.load(gg_ptr + ggb + 1)
    tl.store(gg_ptr + ggb, sx)
    tl.store(gg_ptr + ggb + 1, sy)


def _block_p(n_per_batch: int) -> int:
    # 64 / 256 / 1024 are the tile widths validated on this backend for the
    # grad_input kernel; pick the smallest non-wasteful one.
    if n_per_batch <= 64:
        return 64
    return 256


def grid_sampler_2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
    output_mask: list,
) -> tuple:
    logger.debug(
        "GEMS_KUNLUNXIN GRID_SAMPLER_2D_BACKWARD: mode=%d pad=%d align=%s",
        interpolation_mode,
        padding_mode,
        align_corners,
    )

    if interpolation_mode == 2:
        raise NotImplementedError(
            "grid_sampler_2d_backward: bicubic interpolation is not supported yet"
        )
    if interpolation_mode not in (0, 1) or padding_mode not in (0, 1, 2):
        raise ValueError(
            f"grid_sampler_2d_backward: invalid interpolation_mode "
            f"{interpolation_mode} or padding_mode {padding_mode}"
        )

    N, C, iH, iW = input.shape
    oH, oW = grid.shape[1:3]

    need_grad_input = bool(output_mask[0])
    need_grad_grid = bool(output_mask[1])

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    grid = grid.contiguous()

    grad_input = None
    grad_grid = None
    if need_grad_input:
        grad_input = torch.zeros(
            (N, C, iH, iW), dtype=torch.float32, device=input.device
        )
    if need_grad_grid:
        grad_grid = torch.zeros((N, oH, oW, 2), dtype=torch.float32, device=grid.device)

    n_per_batch = oH * oW
    num_elements = N * n_per_batch
    if num_elements == 0 or C == 0:
        gi = grad_input.to(input.dtype) if need_grad_input else None
        gg = grad_grid.to(grid.dtype) if need_grad_grid else None
        return gi, gg

    if need_grad_input:
        block_p = _block_p(n_per_batch)
        n_tiles = triton.cdiv(n_per_batch, block_p)
        kernel = (
            _grad_input_bilinear_kernel
            if interpolation_mode == 0
            else _grad_input_nearest_kernel
        )
        kernel[(N * C,)](
            grad_output,
            grid,
            grad_input,
            C,
            iH,
            iW,
            oH,
            oW,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_input.stride(0),
            grad_input.stride(1),
            grid.stride(0),
            grid.stride(1),
            grid.stride(2),
            n_per_batch,
            n_tiles,
            padding_mode=padding_mode,
            align_corners=align_corners,
            BLOCK_P=block_p,
        )

    # Nearest interpolation has zero gradient w.r.t. the grid, and grad_grid is
    # already zero-filled, so only bilinear needs the second kernel.
    if need_grad_grid and interpolation_mode == 0:
        block_c = max(_MIN_BLOCK_C, min(_MAX_BLOCK_C, triton.next_power_of_2(C)))
        for c_begin in range(0, C, block_c):
            _grad_grid_bilinear_kernel[(num_elements,)](
                grad_output,
                input,
                grid,
                grad_grid,
                C,
                c_begin,
                min(c_begin + block_c, C),
                iH,
                iW,
                oH,
                oW,
                grad_output.stride(0),
                grad_output.stride(1),
                input.stride(0),
                input.stride(1),
                grid.stride(0),
                grid.stride(1),
                grid.stride(2),
                grad_grid.stride(0),
                grad_grid.stride(1),
                grad_grid.stride(2),
                padding_mode=padding_mode,
                align_corners=align_corners,
                ACCUMULATE=(c_begin > 0),
                BLOCK_C=block_c,
            )

    gi = grad_input.to(input.dtype) if need_grad_input else None
    gg = grad_grid.to(grid.dtype) if need_grad_grid else None
    return gi, gg
