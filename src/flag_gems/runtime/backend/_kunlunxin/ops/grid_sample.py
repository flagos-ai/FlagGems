# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging

import torch
import triton
import triton.language as tl

# Backend registration replaces the top-level symbol; keep direct ops imports consistent.
import flag_gems.ops as _general_ops

logger = logging.getLogger(__name__)


@triton.jit
def _grid_sample_2d_kunlunxin_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    total,
    C: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    IN_STRIDE_N: tl.constexpr,
    IN_STRIDE_C: tl.constexpr,
    IN_STRIDE_H: tl.constexpr,
    IN_STRIDE_W: tl.constexpr,
    GRID_STRIDE_N: tl.constexpr,
    GRID_STRIDE_H: tl.constexpr,
    GRID_STRIDE_W: tl.constexpr,
    GRID_STRIDE_C: tl.constexpr,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    out_x = offsets % W_OUT
    quotient = offsets // W_OUT
    out_y = quotient % H_OUT
    quotient = quotient // H_OUT
    channel = quotient % C
    batch = quotient // C

    grid_base = batch * GRID_STRIDE_N + out_y * GRID_STRIDE_H + out_x * GRID_STRIDE_W
    grid_x = tl.load(grid_ptr + grid_base, mask=mask, other=0.0).to(tl.float32)
    grid_y = tl.load(grid_ptr + grid_base + GRID_STRIDE_C, mask=mask, other=0.0).to(
        tl.float32
    )
    nan_x = ~(tl.abs(grid_x) <= 3.4e38)
    nan_y = ~(tl.abs(grid_y) <= 3.4e38)
    nan_grid = nan_x | nan_y
    grid_x = tl.where(nan_x, -1.0, grid_x)
    grid_y = tl.where(nan_y, -1.0, grid_y)

    if ALIGN_CORNERS:
        source_x = (grid_x + 1.0) * (W_IN - 1) * 0.5
        source_y = (grid_y + 1.0) * (H_IN - 1) * 0.5
    else:
        source_x = ((grid_x + 1.0) * W_IN - 1.0) * 0.5
        source_y = ((grid_y + 1.0) * H_IN - 1.0) * 0.5

    input_base = batch * IN_STRIDE_N + channel * IN_STRIDE_C

    if MODE != 2:
        if PADDING == 1:
            source_x = tl.maximum(0.0, tl.minimum(source_x, W_IN - 1.0))
            source_y = tl.maximum(0.0, tl.minimum(source_y, H_IN - 1.0))
        elif PADDING == 2:
            if ALIGN_CORNERS:
                span_x = tl.maximum(W_IN - 1.0, 1.0)
                span_y = tl.maximum(H_IN - 1.0, 1.0)
                minimum = 0.0
            else:
                span_x = W_IN * 1.0
                span_y = H_IN * 1.0
                minimum = -0.5

            distance_x = tl.abs(source_x - minimum)
            distance_y = tl.abs(source_y - minimum)
            flips_x = tl.floor(distance_x / span_x).to(tl.int32)
            flips_y = tl.floor(distance_y / span_y).to(tl.int32)
            extra_x = distance_x - flips_x.to(tl.float32) * span_x
            extra_y = distance_y - flips_y.to(tl.float32) * span_y
            source_x = tl.where(
                (flips_x & 1) == 0,
                extra_x + minimum,
                span_x - extra_x + minimum,
            )
            source_y = tl.where(
                (flips_y & 1) == 0,
                extra_y + minimum,
                span_y - extra_y + minimum,
            )
            source_x = tl.maximum(0.0, tl.minimum(source_x, W_IN - 1.0))
            source_y = tl.maximum(0.0, tl.minimum(source_y, H_IN - 1.0))

    if MODE == 0:
        floor_x = tl.floor(source_x)
        floor_y = tl.floor(source_y)
        fraction_x = source_x - floor_x
        fraction_y = source_y - floor_y
        floor_x_i = floor_x.to(tl.int32)
        floor_y_i = floor_y.to(tl.int32)
        round_up_x = (fraction_x > 0.5) | ((fraction_x == 0.5) & ((floor_x_i & 1) != 0))
        round_up_y = (fraction_y > 0.5) | ((fraction_y == 0.5) & ((floor_y_i & 1) != 0))
        sample_x = floor_x_i + round_up_x.to(tl.int32)
        sample_y = floor_y_i + round_up_y.to(tl.int32)
        valid = (
            mask
            & ((PADDING != 0) | ~nan_grid)
            & (sample_x >= 0)
            & (sample_x < W_IN)
            & (sample_y >= 0)
            & (sample_y < H_IN)
        )
        sample_x = tl.maximum(0, tl.minimum(sample_x, W_IN - 1))
        sample_y = tl.maximum(0, tl.minimum(sample_y, H_IN - 1))
        value = tl.load(
            input_ptr + input_base + sample_y * IN_STRIDE_H + sample_x * IN_STRIDE_W,
            mask=mask,
            other=0.0,
        )
        value = tl.where(valid, value, 0.0)
    elif MODE == 1:
        x0 = tl.floor(source_x).to(tl.int32)
        y0 = tl.floor(source_y).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = source_x - x0.to(tl.float32)
        wy = source_y - y0.to(tl.float32)

        valid00 = (
            mask
            & ((PADDING != 0) | ~nan_grid)
            & (x0 >= 0)
            & (x0 < W_IN)
            & (y0 >= 0)
            & (y0 < H_IN)
        )
        valid01 = (
            mask
            & ((PADDING != 0) | ~nan_grid)
            & (x1 >= 0)
            & (x1 < W_IN)
            & (y0 >= 0)
            & (y0 < H_IN)
        )
        valid10 = (
            mask
            & ((PADDING != 0) | ~nan_grid)
            & (x0 >= 0)
            & (x0 < W_IN)
            & (y1 >= 0)
            & (y1 < H_IN)
        )
        valid11 = (
            mask
            & ((PADDING != 0) | ~nan_grid)
            & (x1 >= 0)
            & (x1 < W_IN)
            & (y1 >= 0)
            & (y1 < H_IN)
        )

        x0_safe = tl.maximum(0, tl.minimum(x0, W_IN - 1))
        x1_safe = tl.maximum(0, tl.minimum(x1, W_IN - 1))
        y0_safe = tl.maximum(0, tl.minimum(y0, H_IN - 1))
        y1_safe = tl.maximum(0, tl.minimum(y1, H_IN - 1))
        v00 = tl.load(
            input_ptr + input_base + y0_safe * IN_STRIDE_H + x0_safe * IN_STRIDE_W,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v01 = tl.load(
            input_ptr + input_base + y0_safe * IN_STRIDE_H + x1_safe * IN_STRIDE_W,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v10 = tl.load(
            input_ptr + input_base + y1_safe * IN_STRIDE_H + x0_safe * IN_STRIDE_W,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v11 = tl.load(
            input_ptr + input_base + y1_safe * IN_STRIDE_H + x1_safe * IN_STRIDE_W,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v00 = tl.where(valid00, v00, 0.0)
        v01 = tl.where(valid01, v01, 0.0)
        v10 = tl.where(valid10, v10, 0.0)
        v11 = tl.where(valid11, v11, 0.0)
        value = (
            v00 * (1.0 - wx) * (1.0 - wy)
            + v01 * wx * (1.0 - wy)
            + v10 * (1.0 - wx) * wy
            + v11 * wx * wy
        )
    else:
        x_base = tl.floor(source_x).to(tl.int32) - 1
        y_base = tl.floor(source_y).to(tl.int32) - 1
        value = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(4):
            row_value = tl.zeros((BLOCK,), tl.float32)
            sample_y = y_base + j
            distance_y = tl.abs(source_y - sample_y.to(tl.float32))
            weight_y = tl.where(
                distance_y <= 1.0,
                1.25 * distance_y * distance_y * distance_y
                - 2.25 * distance_y * distance_y
                + 1.0,
                tl.where(
                    distance_y < 2.0,
                    -0.75 * distance_y * distance_y * distance_y
                    + 3.75 * distance_y * distance_y
                    - 6.0 * distance_y
                    + 3.0,
                    0.0,
                ),
            )
            for i in tl.static_range(4):
                sample_x = x_base + i
                distance_x = tl.abs(source_x - sample_x.to(tl.float32))
                weight_x = tl.where(
                    distance_x <= 1.0,
                    1.25 * distance_x * distance_x * distance_x
                    - 2.25 * distance_x * distance_x
                    + 1.0,
                    tl.where(
                        distance_x < 2.0,
                        -0.75 * distance_x * distance_x * distance_x
                        + 3.75 * distance_x * distance_x
                        - 6.0 * distance_x
                        + 3.0,
                        0.0,
                    ),
                )

                if PADDING == 1:
                    bounded_x = tl.maximum(0, tl.minimum(sample_x, W_IN - 1))
                    bounded_y = tl.maximum(0, tl.minimum(sample_y, H_IN - 1))
                    valid = mask
                elif PADDING == 2:
                    if ALIGN_CORNERS:
                        span_x = tl.maximum(W_IN - 1.0, 1.0)
                        span_y = tl.maximum(H_IN - 1.0, 1.0)
                        minimum = 0.0
                    else:
                        span_x = W_IN * 1.0
                        span_y = H_IN * 1.0
                        minimum = -0.5
                    distance_ix = tl.abs(sample_x.to(tl.float32) - minimum)
                    distance_iy = tl.abs(sample_y.to(tl.float32) - minimum)
                    flips_ix = tl.floor(distance_ix / span_x).to(tl.int32)
                    flips_iy = tl.floor(distance_iy / span_y).to(tl.int32)
                    extra_ix = distance_ix - flips_ix.to(tl.float32) * span_x
                    extra_iy = distance_iy - flips_iy.to(tl.float32) * span_y
                    bounded_x_float = tl.where(
                        (flips_ix & 1) == 0,
                        extra_ix + minimum,
                        span_x - extra_ix + minimum,
                    )
                    bounded_y_float = tl.where(
                        (flips_iy & 1) == 0,
                        extra_iy + minimum,
                        span_y - extra_iy + minimum,
                    )
                    bounded_x = tl.maximum(
                        0.0, tl.minimum(bounded_x_float, W_IN - 1.0)
                    ).to(tl.int32)
                    bounded_y = tl.maximum(
                        0.0, tl.minimum(bounded_y_float, H_IN - 1.0)
                    ).to(tl.int32)
                    valid = mask
                else:
                    bounded_x = tl.maximum(0, tl.minimum(sample_x, W_IN - 1))
                    bounded_y = tl.maximum(0, tl.minimum(sample_y, H_IN - 1))
                    valid = (
                        mask
                        & ~nan_grid
                        & (sample_x >= 0)
                        & (sample_x < W_IN)
                        & (sample_y >= 0)
                        & (sample_y < H_IN)
                    )

                sample = tl.load(
                    input_ptr
                    + input_base
                    + bounded_y * IN_STRIDE_H
                    + bounded_x * IN_STRIDE_W,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                sample = tl.where(valid, sample, 0.0)
                row_value += sample * weight_x
            value += row_value * weight_y

    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _grid_sample_3d_kunlunxin_kernel(
    output_ptr,
    input_ptr,
    grid_ptr,
    total,
    C: tl.constexpr,
    D_IN: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    D_OUT: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    IN_SN: tl.constexpr,
    IN_SC: tl.constexpr,
    IN_SD: tl.constexpr,
    IN_SH: tl.constexpr,
    IN_SW: tl.constexpr,
    GR_SN: tl.constexpr,
    GR_SD: tl.constexpr,
    GR_SH: tl.constexpr,
    GR_SW: tl.constexpr,
    GR_SC: tl.constexpr,
    MODE: tl.constexpr,
    PADDING: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    ox = offsets % W_OUT
    q = offsets // W_OUT
    oy = q % H_OUT
    q = q // H_OUT
    oz = q % D_OUT
    q = q // D_OUT
    c = q % C
    n = q // C
    base = n * GR_SN + oz * GR_SD + oy * GR_SH + ox * GR_SW
    gx = tl.load(grid_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gy = tl.load(grid_ptr + base + GR_SC, mask=mask, other=0.0).to(tl.float32)
    gz = tl.load(grid_ptr + base + 2 * GR_SC, mask=mask, other=0.0).to(tl.float32)
    nx = ~(tl.abs(gx) <= 3.4e38)
    ny = ~(tl.abs(gy) <= 3.4e38)
    nz = ~(tl.abs(gz) <= 3.4e38)
    nan = nx | ny | nz
    gx, gy, gz = tl.where(nx, -1.0, gx), tl.where(ny, -1.0, gy), tl.where(nz, -1.0, gz)
    if ALIGN:
        x = (gx + 1.0) * (W_IN - 1) * 0.5
        y = (gy + 1.0) * (H_IN - 1) * 0.5
        z = (gz + 1.0) * (D_IN - 1) * 0.5
    else:
        x = ((gx + 1.0) * W_IN - 1.0) * 0.5
        y = ((gy + 1.0) * H_IN - 1.0) * 0.5
        z = ((gz + 1.0) * D_IN - 1.0) * 0.5
    if PADDING == 1:
        x, y, z = (
            tl.maximum(0.0, tl.minimum(x, W_IN - 1.0)),
            tl.maximum(0.0, tl.minimum(y, H_IN - 1.0)),
            tl.maximum(0.0, tl.minimum(z, D_IN - 1.0)),
        )
    elif PADDING == 2:
        if ALIGN:
            lo, sx, sy, sz = (
                0.0,
                tl.maximum(W_IN - 1.0, 1.0),
                tl.maximum(H_IN - 1.0, 1.0),
                tl.maximum(D_IN - 1.0, 1.0),
            )
        else:
            lo, sx, sy, sz = -0.5, W_IN * 1.0, H_IN * 1.0, D_IN * 1.0
        dx, dy, dz = tl.abs(x - lo), tl.abs(y - lo), tl.abs(z - lo)
        fx, fy, fz = (
            tl.floor(dx / sx).to(tl.int32),
            tl.floor(dy / sy).to(tl.int32),
            tl.floor(dz / sz).to(tl.int32),
        )
        ex, ey, ez = (
            dx - fx.to(tl.float32) * sx,
            dy - fy.to(tl.float32) * sy,
            dz - fz.to(tl.float32) * sz,
        )
        x = tl.where((fx & 1) == 0, lo + ex, lo + sx - ex)
        y = tl.where((fy & 1) == 0, lo + ey, lo + sy - ey)
        z = tl.where((fz & 1) == 0, lo + ez, lo + sz - ez)
        x, y, z = (
            tl.maximum(0.0, tl.minimum(x, W_IN - 1.0)),
            tl.maximum(0.0, tl.minimum(y, H_IN - 1.0)),
            tl.maximum(0.0, tl.minimum(z, D_IN - 1.0)),
        )
    base = n * IN_SN + c * IN_SC
    if MODE == 0:
        xf, yf, zf = tl.floor(x), tl.floor(y), tl.floor(z)
        xi, yi, zi = xf.to(tl.int32), yf.to(tl.int32), zf.to(tl.int32)
        xi += ((x - xf > 0.5) | ((x - xf == 0.5) & ((xi & 1) != 0))).to(tl.int32)
        yi += ((y - yf > 0.5) | ((y - yf == 0.5) & ((yi & 1) != 0))).to(tl.int32)
        zi += ((z - zf > 0.5) | ((z - zf == 0.5) & ((zi & 1) != 0))).to(tl.int32)
        valid = (
            mask
            & ((PADDING != 0) | ~nan)
            & (xi >= 0)
            & (xi < W_IN)
            & (yi >= 0)
            & (yi < H_IN)
            & (zi >= 0)
            & (zi < D_IN)
        )
        xi, yi, zi = (
            tl.maximum(0, tl.minimum(xi, W_IN - 1)),
            tl.maximum(0, tl.minimum(yi, H_IN - 1)),
            tl.maximum(0, tl.minimum(zi, D_IN - 1)),
        )
        value = tl.load(
            input_ptr + base + zi * IN_SD + yi * IN_SH + xi * IN_SW,
            mask=mask,
            other=0.0,
        )
        value = tl.where(valid, value, 0.0)
    else:
        x0, y0, z0 = (
            tl.floor(x).to(tl.int32),
            tl.floor(y).to(tl.int32),
            tl.floor(z).to(tl.int32),
        )
        wx, wy, wz = x - x0.to(tl.float32), y - y0.to(tl.float32), z - z0.to(tl.float32)
        value = tl.zeros((BLOCK,), tl.float32)
        for iz in tl.static_range(2):
            for iy in tl.static_range(2):
                for ix in tl.static_range(2):
                    xx, yy, zz = x0 + ix, y0 + iy, z0 + iz
                    valid = (
                        mask
                        & ((PADDING != 0) | ~nan)
                        & (xx >= 0)
                        & (xx < W_IN)
                        & (yy >= 0)
                        & (yy < H_IN)
                        & (zz >= 0)
                        & (zz < D_IN)
                    )
                    xx, yy, zz = (
                        tl.maximum(0, tl.minimum(xx, W_IN - 1)),
                        tl.maximum(0, tl.minimum(yy, H_IN - 1)),
                        tl.maximum(0, tl.minimum(zz, D_IN - 1)),
                    )
                    v = tl.load(
                        input_ptr + base + zz * IN_SD + yy * IN_SH + xx * IN_SW,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    value += (
                        tl.where(valid, v, 0.0)
                        * tl.where(ix == 0, 1.0 - wx, wx)
                        * tl.where(iy == 0, 1.0 - wy, wy)
                        * tl.where(iz == 0, 1.0 - wz, wz)
                    )
    tl.store(output_ptr + offsets, value, mask=mask)


def grid_sample(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN GRID SAMPLE")
    from flag_gems.ops.grid_sample import _validate_grid_sample_input

    _validate_grid_sample_input(input, grid, mode, padding_mode)
    if input.dim() == 5:
        n, c, _, _, _ = input.shape
        _, d_out, h_out, w_out, _ = grid.shape
        output = torch.empty(
            (n, c, d_out, h_out, w_out), dtype=input.dtype, device=input.device
        )
        total = output.numel()
        if total == 0:
            return output
        _grid_sample_3d_kunlunxin_kernel[(triton.cdiv(total, 64),)](
            output,
            input,
            grid,
            total,
            C=c,
            D_IN=input.shape[2],
            H_IN=input.shape[3],
            W_IN=input.shape[4],
            D_OUT=d_out,
            H_OUT=h_out,
            W_OUT=w_out,
            IN_SN=input.stride(0),
            IN_SC=input.stride(1),
            IN_SD=input.stride(2),
            IN_SH=input.stride(3),
            IN_SW=input.stride(4),
            GR_SN=grid.stride(0),
            GR_SD=grid.stride(1),
            GR_SH=grid.stride(2),
            GR_SW=grid.stride(3),
            GR_SC=grid.stride(4),
            MODE={"nearest": 0, "bilinear": 1}[mode],
            PADDING={"zeros": 0, "border": 1, "reflection": 2}[padding_mode],
            ALIGN=align_corners,
            BLOCK=64,
            num_warps=4,
            num_stages=1,
            isCloseVectorization=True,
            buffer_size_limit=2048,
            unroll_num=8,
        )
        return output

    n, c, _, _ = input.shape
    _, h_out, w_out, _ = grid.shape
    output = torch.empty((n, c, h_out, w_out), dtype=input.dtype, device=input.device)
    total = output.numel()
    if total == 0:
        return output

    mode_id = {"nearest": 0, "bilinear": 1, "bicubic": 2}[mode]
    padding_id = {"zeros": 0, "border": 1, "reflection": 2}[padding_mode]
    block = 64
    _grid_sample_2d_kunlunxin_kernel[(triton.cdiv(total, block),)](
        output,
        input,
        grid,
        total,
        C=c,
        H_IN=input.shape[2],
        W_IN=input.shape[3],
        H_OUT=h_out,
        W_OUT=w_out,
        IN_STRIDE_N=input.stride(0),
        IN_STRIDE_C=input.stride(1),
        IN_STRIDE_H=input.stride(2),
        IN_STRIDE_W=input.stride(3),
        GRID_STRIDE_N=grid.stride(0),
        GRID_STRIDE_H=grid.stride(1),
        GRID_STRIDE_W=grid.stride(2),
        GRID_STRIDE_C=grid.stride(3),
        MODE=mode_id,
        PADDING=padding_id,
        ALIGN_CORNERS=align_corners,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
        isCloseVectorization=True,
        buffer_size_limit=2048,
        unroll_num=8,
    )
    return output


_general_ops.grid_sample = grid_sample
