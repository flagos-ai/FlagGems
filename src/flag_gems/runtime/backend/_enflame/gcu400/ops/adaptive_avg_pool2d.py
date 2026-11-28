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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Tuning knobs, adjusted one variable at a time during the optimization rounds.
# TILE_LIMIT caps the uniform-arm staging tile: the reshape chain bills ~32B
# of DSM per tile element on GCU400 (a 64K-element tile needs 2MB against the
# 917504B hardware limit), so 16384 elements is the largest safe power of two.
# GRID_LIMIT keeps the persistent walk inside one <=48-CTA wave.
# UNIFORM_WARPS > 0 pins the uniform-arm warp count, bypassing the tile rule.
POOL_TILE_ELEMENT_LIMIT = 16384
POOL_GRID_LIMIT = 48
POOL_UNIFORM_WARPS = 0
POOL_GENERAL_BLOCK_OUT = 64
POOL_GENERAL_TILE_LIMIT = 16384
POOL_GENERAL_BLOCK_W_CAP = 8192


@libentry()
@triton.jit
def _adaptive_pool2d_uniform_kernel(
    X,
    Y,
    NC,
    IH,
    IW,
    OH,
    OW,
    P,
    R_W,
    COL_BLOCKS,
    UNITS,
    INV_HW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    OUT_SPAN: tl.constexpr,
    ROWS_PER_SLAB: tl.constexpr,
):
    """Fast arm for IH % OH == 0, IW % OW == 0 and power-of-two KH/KW.

    Every plane is exactly OH*KH input rows by OW*KW input columns, so one
    program can slab P consecutive planes at once: relative input row r is
    p*IH + oh*KH + kh and the loaded rectangle stays a plain affine 2D tile
    (row stride IW, contiguous columns) that takes the vectorized tar_load
    path. The reductions over kh and kw are value-side reshapes, so every
    tensor stays at rank <= 3 (the 3D pointer-array tile fails the GCU400
    PassManager) and the fp32 accumulation happens in a single pass with the
    host-precomputed 1/(KH*KW) reciprocal. A grid-stride walk keeps the grid
    inside one <=48-CTA wave regardless of N*C.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for unit in tl.range(pid, UNITS, num_progs):
        plane_unit = unit // COL_BLOCKS
        col_block = unit % COL_BLOCKS
        p0 = plane_unit * P
        p_valid = tl.minimum(P, NC - p0)
        rows = tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        row = p0 * IH + rows
        col = col_block * (R_W * KW) + cols
        col_limit = tl.minimum(R_W * KW, IW - col_block * R_W * KW)
        valid = (rows < p_valid * IH)[:, None] & (cols < col_limit)[None, :]
        values = tl.load(
            X + row[:, None] * IW + col[None, :], mask=valid, other=0.0
        ).to(tl.float32)
        # Column reduction: (BLOCK_ROWS, BLOCK_COLS) -> (.., OUT_SPAN, KW).
        part = tl.sum(tl.reshape(values, (BLOCK_ROWS, OUT_SPAN, KW)), 2)
        # Row reduction: rows relabel to (p*OH + oh, kh) across the slab.
        mean = tl.sum(tl.reshape(part, (ROWS_PER_SLAB, KH, OUT_SPAN)), 1)
        mean = mean * INV_HW
        out_rows = tl.arange(0, ROWS_PER_SLAB)
        out_cols = tl.arange(0, OUT_SPAN)
        out_col = col_block * R_W + out_cols
        # A unit owns exactly R_W output columns; when the pow2 OUT_SPAN pad
        # is wider (shrink loop) the extra lanes would collide with the next
        # unit's territory, so clamp to this unit's true column count.
        rw_true = tl.minimum(R_W, OW - col_block * R_W)
        tl.store(
            Y + p0 * (OH * OW) + (out_rows * OW)[:, None] + out_col[None, :],
            mean.to(Y.dtype.element_ty),
            mask=(out_rows < p_valid * OH)[:, None] & (out_cols < rw_true)[None, :],
        )


@libentry()
@triton.jit
def _adaptive_pool2d_general_kernel(
    X,
    Y,
    IH,
    IW,
    OH,
    OW,
    BLOCKS_PER_PLANE,
    UNITS,
    KW: tl.constexpr,
    MAX_WINDOW: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """General arm: aten bin edges floor(i*I/O) .. ceil((i+1)*I/O).

    One program owns BLOCK_OUT output positions of a single (N, C) plane and
    walks units with a grid-stride loop, so the grid stays inside one <=48-CTA
    wave (GCU400 also caps grid.x at 65535). The 2D (BLOCK_OUT, BLOCK_W) tile
    replaces the one-output-per-iteration serial loop: all bins reduce in one
    tl.sum instead of one reduction tree per output element. Windows wider
    than BLOCK_W fall back to a chunked accumulation loop, which folds away
    when the window fits in a single chunk.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for unit in tl.range(pid, UNITS, num_progs):
        out_block = unit % BLOCKS_PER_PLANE
        plane = unit // BLOCKS_PER_PLANE
        out_pos = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        oh = out_pos // OW
        ow = out_pos - oh * OW
        pos_ok = out_pos < OH * OW
        bin_h_start = (oh * IH) // OH
        bin_h_end = ((oh + 1) * IH + OH - 1) // OH
        bin_w_start = (ow * IW) // OW
        bin_w_end = ((ow + 1) * IW + OW - 1) // OW
        win_h = tl.maximum(bin_h_end - bin_h_start, 0)
        win_w = tl.maximum(bin_w_end - bin_w_start, 0)
        accumulator = tl.zeros((BLOCK_OUT, BLOCK_W), tl.float32)
        for start in range(0, MAX_WINDOW, BLOCK_W):
            window_index = start + tl.arange(0, BLOCK_W)
            h_off = window_index // KW
            w_off = window_index - h_off * KW
            h = bin_h_start[:, None] + h_off[None, :]
            w = bin_w_start[:, None] + w_off[None, :]
            valid = (
                pos_ok[:, None]
                & (h_off[None, :] < win_h[:, None])
                & (w_off[None, :] < win_w[:, None])
            )
            values = tl.load(
                X + plane * (IH * IW) + h * IW + w, mask=valid, other=0.0
            ).to(tl.float32)
            accumulator += values
        count = tl.maximum(win_h * win_w, 1).to(tl.float32)
        mean = tl.sum(accumulator, 1) / count
        tl.store(
            Y + plane * (OH * OW) + out_pos,
            mean.to(Y.dtype.element_ty),
            mask=pos_ok,
        )


def _normalize_output_size_2d(output_size):
    if isinstance(output_size, int):
        return output_size, output_size
    oh, ow = output_size
    return oh, ow


def _is_pow2(value):
    return value > 0 and (value & (value - 1)) == 0


def _launch_uniform(input: torch.Tensor, output: torch.Tensor, nc, ih, iw, oh, ow):
    kh = ih // oh
    kw = iw // ow
    r_w = ow
    block_cols = triton.next_power_of_2(r_w * kw)
    block_rows = min(
        triton.next_power_of_2(nc * ih), POOL_TILE_ELEMENT_LIMIT // block_cols
    )
    while block_rows < ih and r_w > 1:
        r_w = max(1, r_w // 2)
        block_cols = triton.next_power_of_2(r_w * kw)
        block_rows = min(
            triton.next_power_of_2(nc * ih), POOL_TILE_ELEMENT_LIMIT // block_cols
        )
    if block_rows < ih:
        return False
    planes = max(1, block_rows // ih)
    out_span = block_cols // kw
    rows_per_slab = block_rows // kh
    col_blocks = triton.cdiv(ow, r_w)
    units = triton.cdiv(nc, planes) * col_blocks
    grid = (min(units, POOL_GRID_LIMIT),)
    # Measured on GCU400/L600 (Triton GCU 4.1.6): every 2D tile op pays a
    # ~0.2-0.8us DSM round trip regardless of tile size, so the largest tile
    # that fits the per-CTA DSM budget amortizes best. num_warps=1 wins
    # uniformly here -- nw=2/4 regressed every core shape 2-8x -- and the
    # 16K-element tile keeps ~8 live buffers inside the 917504B DSM limit
    # (a 64K tile bills ~32B/element and exceeds it).
    if POOL_UNIFORM_WARPS > 0:
        num_warps = POOL_UNIFORM_WARPS
    else:
        num_warps = 1
    _adaptive_pool2d_uniform_kernel[grid](
        input,
        output,
        nc,
        ih,
        iw,
        oh,
        ow,
        planes,
        r_w,
        col_blocks,
        units,
        1.0 / (kh * kw),
        KH=kh,
        KW=kw,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        OUT_SPAN=out_span,
        ROWS_PER_SLAB=rows_per_slab,
        num_warps=num_warps,
    )
    return True


def _launch_general(input: torch.Tensor, output: torch.Tensor, nc, ih, iw, oh, ow):
    kh = triton.cdiv(ih, oh) + (1 if ih % oh else 0)
    kw = triton.cdiv(iw, ow) + (1 if iw % ow else 0)
    max_window = kh * kw
    block_w = min(triton.next_power_of_2(max_window), POOL_GENERAL_BLOCK_W_CAP)
    block_out = min(
        POOL_GENERAL_BLOCK_OUT,
        triton.next_power_of_2(oh * ow),
        max(1, POOL_GENERAL_TILE_LIMIT // block_w),
    )
    blocks_per_plane = triton.cdiv(oh * ow, block_out)
    units = blocks_per_plane * nc
    grid = (min(units, POOL_GRID_LIMIT),)
    _adaptive_pool2d_general_kernel[grid](
        input,
        output,
        ih,
        iw,
        oh,
        ow,
        blocks_per_plane,
        units,
        KW=kw,
        MAX_WINDOW=max_window,
        BLOCK_OUT=block_out,
        BLOCK_W=block_w,
        num_warps=1,
    )


def _launch_adaptive_pool_windows(input: torch.Tensor, output: torch.Tensor):
    """Pool a contiguous ``input`` into a contiguous ``output`` in one launch.

    ``output`` must be contiguous: the kernels address it with flat offsets,
    so a strided buffer would be written at the wrong positions. The uniform
    arm (exact division along both axes, power-of-two windows) slabs whole
    planes through an affine tile; everything else falls back to the general
    bin-edge kernel, which keeps aten's per-window floor/ceil semantics for
    non-divisible and upsampling sizes.
    """
    n, c, ih, iw = input.shape
    oh, ow = output.shape[-2:]
    nc = n * c
    with torch_device_fn.device(input.device):
        if (
            ih % oh == 0
            and iw % ow == 0
            and _is_pow2(ih // oh)
            and _is_pow2(iw // ow)
            and input.numel() < 2**31
            and output.numel() < 2**31
            and _launch_uniform(input, output, nc, ih, iw, oh, ow)
        ):
            return
        _launch_general(input, output, nc, ih, iw, oh, ow)


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS_ENFLAME ADAPTIVE_AVG_POOL2D")
    input = input.contiguous()
    oh, ow = _normalize_output_size_2d(output_size)
    n, c, ih, iw = input.shape
    output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if output.numel() == 0 or ih == 0 or iw == 0:
        return output
    _launch_adaptive_pool_windows(input, output)
    return output


def _adaptive_avg_pool2d_out(input: torch.Tensor, output_size, *, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME _ADAPTIVE_AVG_POOL2D_OUT")
    if out.dtype != input.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {input.dtype}, "
            f"but got {out.dtype} instead"
        )
    input = input.contiguous()
    oh, ow = _normalize_output_size_2d(output_size)
    n, c, ih, iw = input.shape
    expected_shape = (n, c, oh, ow)
    # ATen resizes a mismatched out tensor instead of raising, so mirror that.
    if tuple(out.shape) != expected_shape:
        out.resize_(expected_shape)
    if out.numel() == 0 or ih == 0 or iw == 0:
        return out
    # The kernel writes through flat offsets, so a strided out tensor has to be
    # filled via a contiguous staging buffer. ``empty_like`` would preserve a
    # channels_last layout, which is still not contiguous, so request the
    # contiguous format explicitly.
    buffer = (
        out
        if out.is_contiguous()
        else torch.empty_like(out, memory_format=torch.contiguous_format)
    )
    _launch_adaptive_pool_windows(input, buffer)
    if buffer is not out:
        out.copy_(buffer)
    return out
