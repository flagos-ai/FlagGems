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
POOL_BLOCK_OUT_LIMIT = 64
POOL_TILE_ELEMENT_LIMIT = 16384
POOL_NUM_WARPS = 4
# 单 pass 候选：直接对矩形窗口求均值，省掉 staging buffer 与第二次 launch。
POOL_USE_SINGLE_PASS = 1
POOL_SINGLE_BLOCK_R = 2048
POOL_SINGLE_GRID_LIMIT = 65535


@libentry()
@triton.jit
def _adaptive_pool2d_pass_kernel(
    X,
    Y,
    L_IN,
    L_OUT,
    PLANE_SPAN,
    X_NC_STRIDE,
    X_COL_STRIDE,
    X_POOL_STRIDE,
    Y_NC_STRIDE,
    Y_COL_STRIDE,
    Y_POOL_STRIDE,
    BLOCKS_PER_PLANE,
    MAX_WINDOW: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    """Average one spatial axis of a contiguous (NC, axis, column) tensor.

    Adaptive average pooling over a rectangular window is separable: averaging
    along W and then along H equals averaging the whole rectangle at once. Each
    program owns BLOCK_OUT output bins of a single plane, so the load tile stays
    a compact 2D tile instead of the BO x KH x KW pointer array that makes the
    GCU400 PassManager fail. aten's bin edges are floor(i*L_IN/L_OUT) and
    ceil((i+1)*L_IN/L_OUT); MAX_WINDOW covers the largest possible bin.
    """
    pid = tl.program_id(0)
    out_block = pid % BLOCKS_PER_PLANE
    plane = pid // BLOCKS_PER_PLANE
    nc = plane // PLANE_SPAN
    col = plane % PLANE_SPAN

    out_pos = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    bin_start = (out_pos * L_IN) // L_OUT
    bin_end = ((out_pos + 1) * L_IN + L_OUT - 1) // L_OUT
    window = tl.arange(0, MAX_WINDOW)
    in_pos = bin_start[:, None] + window[None, :]
    valid = (out_pos[:, None] < L_OUT) & (in_pos < bin_end[:, None]) & (in_pos < L_IN)
    x_base = nc * X_NC_STRIDE + col * X_COL_STRIDE
    y_base = nc * Y_NC_STRIDE + col * Y_COL_STRIDE
    values = tl.load(X + x_base + in_pos * X_POOL_STRIDE, mask=valid, other=0.0).to(
        tl.float32
    )
    count = tl.maximum(bin_end - bin_start, 1)
    mean = tl.sum(values, 1) / count.to(tl.float32)
    tl.store(
        Y + y_base + out_pos * Y_POOL_STRIDE,
        mean.to(Y.dtype.element_ty),
        mask=out_pos < L_OUT,
    )


def _run_pool_pass(
    source,
    target,
    length_in,
    length_out,
    plane_span,
    num_planes,
    x_nc_stride,
    x_col_stride,
    x_pool_stride,
    y_nc_stride,
    y_col_stride,
    y_pool_stride,
    num_warps,
):
    """Launch one separable pass described by explicit memory strides."""
    max_window = triton.next_power_of_2(triton.cdiv(length_in, length_out) + 1)
    block_out = min(
        POOL_BLOCK_OUT_LIMIT,
        triton.next_power_of_2(length_out),
        max(1, POOL_TILE_ELEMENT_LIMIT // max_window),
    )
    blocks_per_plane = triton.cdiv(length_out, block_out)
    grid = (blocks_per_plane * num_planes * plane_span,)
    _adaptive_pool2d_pass_kernel[grid](
        source,
        target,
        length_in,
        length_out,
        plane_span,
        x_nc_stride,
        x_col_stride,
        x_pool_stride,
        y_nc_stride,
        y_col_stride,
        y_pool_stride,
        blocks_per_plane,
        MAX_WINDOW=max_window,
        BLOCK_OUT=block_out,
        num_warps=num_warps,
    )


@libentry()
@triton.jit
def _adaptive_pool2d_single_kernel(
    X,
    Y,
    IH,
    IW,
    OH,
    OW,
    NUM_OUT,
    GRID_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Single-kernel rectangle reduction over a 1D window tile.

    One program covers one or more output elements via a grid stride loop; the
    window index r maps to (row, column) inside the aten bin. Keeping the tile
    one-dimensional avoids the huge BO x KH x KW pointer array that fails the
    GCU400 PassManager, and removes the intermediate fp32 staging tensor.
    """
    for out_index in tl.range(tl.program_id(0), NUM_OUT, GRID_SIZE):
        nc = out_index // (OH * OW)
        oh = (out_index // OW) % OH
        ow = out_index % OW
        bin_h_start = (oh * IH) // OH
        bin_h_end = ((oh + 1) * IH + OH - 1) // OH
        bin_w_start = (ow * IW) // OW
        bin_w_end = ((ow + 1) * IW + OW - 1) // OW
        window_height = bin_h_end - bin_h_start
        window_width = bin_w_end - bin_w_start
        window_elements = tl.maximum(window_height * window_width, 1)
        offsets = tl.arange(0, BLOCK_R)
        accumulator = tl.zeros((BLOCK_R,), tl.float32)
        for start in range(0, window_elements, BLOCK_R):
            window_index = start + offsets
            row = bin_h_start + window_index // window_width
            column = bin_w_start + window_index % window_width
            values = tl.load(
                X + nc * IH * IW + row * IW + column,
                mask=window_index < window_elements,
                other=0.0,
            ).to(tl.float32)
            accumulator += values
        mean = tl.sum(accumulator, 0) / window_elements.to(tl.float32)
        tl.store(Y + out_index, mean.to(Y.dtype.element_ty))


def _normalize_output_size_2d(output_size):
    if isinstance(output_size, int):
        return output_size, output_size
    oh, ow = output_size
    return oh, ow


def _launch_adaptive_pool_windows(input: torch.Tensor, output: torch.Tensor):
    """Pool a contiguous ``input`` into a contiguous ``output`` in two passes.

    ``output`` must be contiguous: both passes address it with flat offsets, so
    a strided buffer would be written at the wrong positions. The W pass writes
    an fp32 staging tensor to keep the separable reduction free of intermediate
    rounding; the H pass converts back to the caller's dtype on store.
    """
    n, c, ih, iw = input.shape
    oh, ow = output.shape[-2:]
    num_planes = n * c
    with torch_device_fn.device(input.device):
        if POOL_USE_SINGLE_PASS:
            # 结构性候选：一次 launch 直接完成矩形窗口规约。
            grid_size = min(POOL_SINGLE_GRID_LIMIT, output.numel())
            _adaptive_pool2d_single_kernel[(grid_size,)](
                input,
                output,
                ih,
                iw,
                oh,
                ow,
                output.numel(),
                GRID_SIZE=grid_size,
                BLOCK_R=POOL_SINGLE_BLOCK_R,
                num_warps=POOL_NUM_WARPS,
            )
            return
        row_staging = torch.empty(
            (n, c, ih, ow), device=input.device, dtype=torch.float32
        )
        # Pass 1: (NC, H, W) -> (NC, H, OW), pooling the innermost W axis.
        _run_pool_pass(
            input,
            row_staging,
            iw,
            ow,
            ih,
            num_planes,
            ih * iw,
            iw,
            1,
            ih * ow,
            ow,
            1,
            POOL_NUM_WARPS,
        )
        # Pass 2: (NC, H, OW) -> (NC, OH, OW), pooling H with stride OW.
        _run_pool_pass(
            row_staging,
            output,
            ih,
            oh,
            ow,
            num_planes,
            ih * ow,
            1,
            ow,
            oh * ow,
            1,
            ow,
            POOL_NUM_WARPS,
        )


def adaptive_avg_pool2d(input: torch.Tensor, output_size):
    logger.debug("GEMS ADAPTIVE_AVG_POOL2D")
    input = input.contiguous()
    oh, ow = _normalize_output_size_2d(output_size)
    n, c, ih, iw = input.shape
    output = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    if output.numel() == 0 or ih == 0 or iw == 0:
        return output
    _launch_adaptive_pool_windows(input, output)
    return output


def _adaptive_avg_pool2d_out(input: torch.Tensor, output_size, *, out: torch.Tensor):
    logger.debug("GEMS _ADAPTIVE_AVG_POOL2D_OUT")
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
