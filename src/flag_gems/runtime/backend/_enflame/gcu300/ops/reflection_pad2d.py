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
#
# GCU300-specific reflection_pad2d.
# Hybrid strategy based on output size:
#   - Small shapes (<500K elements): single full gather kernel (1 launch)
#   - Large shapes (>=500K elements): center DMA + edge rows DMA + edge cols gather
# The split avoids GCU300 compiler timeout from complex boolean masks while
# using block_ptr DMA fast path for the 97%+ center region on large outputs.

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_SMALL_SHAPE_THRESHOLD = 500_000


@triton.jit
def reflection_pad2d_gather_kernel(
    in_ptr,
    out_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    total_elems,
    GRID_DIM: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Single-kernel full gather: processes all output elements with reflection."""
    pid = tl.program_id(axis=0)
    HW = H_out * W_out
    Hm1 = H_in - 1
    Wm1 = W_in - 1
    pH = 2 * Hm1
    pW = 2 * Wm1

    for block_id in tl.range(pid, tl.cdiv(total_elems, BLOCK_HW), GRID_DIM):
        offs = block_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs < total_elems

        pid_b = offs // HW
        rem = offs % HW
        h_idx = rem // W_out
        w_idx = rem % W_out

        base_in = pid_b * (H_in * W_in)

        y = h_idx.to(tl.int32) - pad_top
        t_h = tl.abs(y)
        m_h = t_h % pH
        ih = tl.where(m_h < H_in, m_h, pH - m_h)

        x = w_idx.to(tl.int32) - pad_left
        t_w = tl.abs(x)
        m_w = t_w % pW
        iw = tl.where(m_w < W_in, m_w, pW - m_w)

        in_offs = base_in + ih * W_in + iw
        vals = tl.load(in_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def center_dma_kernel(
    in_ptr,
    out_ptr,
    B,
    H_in,
    W_in,
    pad_top,
    pad_left,
    H_out,
    W_out,
    GRID_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Copy center region via 2D block_ptr DMA."""
    pid = tl.program_id(axis=0)
    n_tiles_h = tl.cdiv(H_in, BLOCK_H)
    n_tiles_w = tl.cdiv(W_in, BLOCK_W)
    total_tiles = B * n_tiles_h * n_tiles_w
    HW_in = H_in * W_in
    HW_out = H_out * W_out

    for tile_id in tl.range(pid, total_tiles, GRID_DIM):
        pid_b = tile_id // (n_tiles_h * n_tiles_w)
        rem = tile_id % (n_tiles_h * n_tiles_w)
        pid_h = rem // n_tiles_w
        pid_w = rem % n_tiles_w

        in_base = in_ptr + pid_b * HW_in
        out_base = out_ptr + pid_b * HW_out + pad_top * W_out + pad_left

        in_block = tl.make_block_ptr(
            base=in_base,
            shape=(H_in, W_in),
            strides=(W_in, 1),
            offsets=(pid_h * BLOCK_H, pid_w * BLOCK_W),
            block_shape=(BLOCK_H, BLOCK_W),
            order=(1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_base,
            shape=(H_in, W_in),
            strides=(W_out, 1),
            offsets=(pid_h * BLOCK_H, pid_w * BLOCK_W),
            block_shape=(BLOCK_H, BLOCK_W),
            order=(1, 0),
        )
        vals = tl.load(in_block, boundary_check=(0, 1), padding_option="zero")
        tl.store(out_block, vals, boundary_check=(0, 1))


@triton.jit
def edge_rows_dma_kernel(
    in_ptr,
    out_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    h_start,
    h_count,
    GRID_DIM: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Row-by-row 1D block_ptr DMA for edge rows (top/bottom), center columns.
    h is reflected per row; w is identity (contiguous within each row)."""
    pid = tl.program_id(axis=0)
    total_rows = B * h_count
    HW_in = H_in * W_in
    HW_out = H_out * W_out
    Hm1 = H_in - 1
    pH = 2 * Hm1
    n_tiles_w = tl.cdiv(W_in, BLOCK_W)
    total_tiles = total_rows * n_tiles_w

    for tile_id in tl.range(pid, total_tiles, GRID_DIM):
        pid_row = tile_id // n_tiles_w
        pid_w = tile_id % n_tiles_w

        pid_b = pid_row // h_count
        h_local = pid_row % h_count
        h_out = h_start + h_local

        y = h_out.to(tl.int32) - pad_top
        t_h = tl.abs(y)
        m_h = t_h % pH
        ih = tl.where(m_h < H_in, m_h, pH - m_h)

        in_base = in_ptr + pid_b * HW_in + ih * W_in
        out_base = out_ptr + pid_b * HW_out + h_out * W_out + pad_left

        in_block = tl.make_block_ptr(
            base=in_base,
            shape=(W_in,),
            strides=(1,),
            offsets=(pid_w * BLOCK_W,),
            block_shape=(BLOCK_W,),
            order=(0,),
        )
        out_block = tl.make_block_ptr(
            base=out_base,
            shape=(W_in,),
            strides=(1,),
            offsets=(pid_w * BLOCK_W,),
            block_shape=(BLOCK_W,),
            order=(0,),
        )
        vals = tl.load(in_block, boundary_check=(0,), padding_option="zero")
        tl.store(out_block, vals, boundary_check=(0,))


@triton.jit
def edge_gather_kernel(
    in_ptr,
    out_ptr,
    B,
    H_in,
    W_in,
    pad_left,
    pad_top,
    H_out,
    W_out,
    h_start,
    h_count,
    w_start,
    w_count,
    GRID_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-element gather for one rectangular edge region with reflection."""
    pid = tl.program_id(axis=0)
    elems_per_batch = h_count * w_count
    total = B * elems_per_batch
    HW_in = H_in * W_in
    HW_out = H_out * W_out
    Hm1 = H_in - 1
    Wm1 = W_in - 1
    pH = 2 * Hm1
    pW = 2 * Wm1

    for block_id in tl.range(pid, tl.cdiv(total, BLOCK), GRID_DIM):
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        pid_b = offs // elems_per_batch
        rem = offs % elems_per_batch
        h_local = rem // w_count
        w_local = rem % w_count
        h_out = h_start + h_local
        w_out = w_start + w_local

        y = h_out.to(tl.int32) - pad_top
        t_h = tl.abs(y)
        m_h = t_h % pH
        ih = tl.where(m_h < H_in, m_h, pH - m_h)

        x = w_out.to(tl.int32) - pad_left
        t_w = tl.abs(x)
        m_w = t_w % pW
        iw = tl.where(m_w < W_in, m_w, pW - m_w)

        in_offs = pid_b * HW_in + ih * W_in + iw
        out_offs = pid_b * HW_out + h_out * W_out + w_out
        vals = tl.load(in_ptr + in_offs, mask=mask, other=0)
        tl.store(out_ptr + out_offs, vals, mask=mask)


def launch_reflection_pad2d(input, padding, out=None):
    if not isinstance(padding, (list, tuple)):
        raise ValueError("padding must be a sequence")
    if len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of length 4: "
            "(pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = [int(p) for p in padding]

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("padding values must be >= 0")

    if input.dim() < 3:
        raise ValueError("input must have at least 3 dimensions")

    x = input.contiguous()
    H_in = int(x.shape[-2])
    W_in = int(x.shape[-1])
    if H_in < 2 or W_in < 2:
        raise ValueError(
            "input spatial dimensions must be at least 2 "
            "for reflection padding when padding > 0"
        )
    if pad_left >= W_in or pad_right >= W_in or pad_top >= H_in or pad_bottom >= H_in:
        raise ValueError(
            "padding values must be less than the input spatial dimensions "
            "for reflection padding"
        )

    H_out = H_in + pad_top + pad_bottom
    W_out = W_in + pad_left + pad_right

    leading_shape = x.shape[:-2]
    B = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

    if out is None:
        out = torch.empty(
            (*leading_shape, H_out, W_out), device=x.device, dtype=x.dtype
        )
    else:
        expected_shape = (*leading_shape, H_out, W_out)
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"out tensor has shape {tuple(out.shape)}, expected {expected_shape}"
            )
        if out.dtype != x.dtype:
            raise ValueError(
                f"out dtype {out.dtype} does not match input dtype {x.dtype}"
            )
        if out.device != x.device:
            raise ValueError("out must be on the same device as input")
        out = out.contiguous()

    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        out.copy_(x)
        return out

    GRID_DIM = 24
    total_elems = B * H_out * W_out

    if total_elems < _SMALL_SHAPE_THRESHOLD:
        # Small shape: single full gather kernel (1 launch, low overhead)
        BLOCK_HW = 8192
        num_blocks = triton.cdiv(total_elems, BLOCK_HW)
        grid = (min(GRID_DIM, num_blocks),)
        reflection_pad2d_gather_kernel[grid](
            x,
            out,
            B,
            H_in,
            W_in,
            pad_left,
            pad_top,
            H_out,
            W_out,
            total_elems,
            GRID_DIM=min(GRID_DIM, num_blocks),
            BLOCK_HW=BLOCK_HW,
        )
    else:
        # Large shape: center DMA + edge rows DMA + edge cols gather
        BLOCK_W = min(256, triton.next_power_of_2(W_in))
        BLOCK_H = 64

        # 1. Center DMA: center rows x center cols (2D block_ptr)
        n_tiles_h = triton.cdiv(H_in, BLOCK_H)
        n_tiles_w = triton.cdiv(W_in, BLOCK_W)
        center_tiles = B * n_tiles_h * n_tiles_w
        center_grid = (min(GRID_DIM, center_tiles),)
        center_dma_kernel[center_grid](
            x,
            out,
            B,
            H_in,
            W_in,
            pad_top,
            pad_left,
            H_out,
            W_out,
            GRID_DIM=min(GRID_DIM, center_tiles),
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            num_warps=1,
        )

        # 2. Edge rows center columns: DMA (h reflected, w contiguous)
        if pad_top > 0:
            edge_tiles = B * pad_top * n_tiles_w
            edge_grid = (min(GRID_DIM, edge_tiles),)
            edge_rows_dma_kernel[edge_grid](
                x,
                out,
                B,
                H_in,
                W_in,
                pad_left,
                pad_top,
                H_out,
                W_out,
                0,
                pad_top,
                GRID_DIM=min(GRID_DIM, edge_tiles),
                BLOCK_W=BLOCK_W,
                num_warps=1,
            )
        if pad_bottom > 0:
            edge_tiles = B * pad_bottom * n_tiles_w
            edge_grid = (min(GRID_DIM, edge_tiles),)
            edge_rows_dma_kernel[edge_grid](
                x,
                out,
                B,
                H_in,
                W_in,
                pad_left,
                pad_top,
                H_out,
                W_out,
                pad_top + H_in,
                pad_bottom,
                GRID_DIM=min(GRID_DIM, edge_tiles),
                BLOCK_W=BLOCK_W,
                num_warps=1,
            )

        # 3. Left/right padding columns (ALL rows including corners)
        BLOCK_EDGE = 8192
        if pad_left > 0:
            edge_elems = B * H_out * pad_left
            edge_grid = (min(GRID_DIM, triton.cdiv(edge_elems, BLOCK_EDGE)),)
            edge_gather_kernel[edge_grid](
                x,
                out,
                B,
                H_in,
                W_in,
                pad_left,
                pad_top,
                H_out,
                W_out,
                0,
                H_out,
                0,
                pad_left,
                GRID_DIM=min(GRID_DIM, triton.cdiv(edge_elems, BLOCK_EDGE)),
                BLOCK=BLOCK_EDGE,
            )
        if pad_right > 0:
            edge_elems = B * H_out * pad_right
            edge_grid = (min(GRID_DIM, triton.cdiv(edge_elems, BLOCK_EDGE)),)
            edge_gather_kernel[edge_grid](
                x,
                out,
                B,
                H_in,
                W_in,
                pad_left,
                pad_top,
                H_out,
                W_out,
                0,
                H_out,
                pad_left + W_in,
                pad_right,
                GRID_DIM=min(GRID_DIM, triton.cdiv(edge_elems, BLOCK_EDGE)),
                BLOCK=BLOCK_EDGE,
            )

    return out


def reflection_pad2d(input, padding):
    logger.debug("GEMS_ENFLAME_GCU300 REFLECTION_PAD2D")
    return launch_reflection_pad2d(input, padding, out=None)


def reflection_pad2d_out(input, padding, out):
    logger.debug("GEMS_ENFLAME_GCU300 REFLECTION_PAD2D_OUT")
    return launch_reflection_pad2d(input, padding, out=out)
