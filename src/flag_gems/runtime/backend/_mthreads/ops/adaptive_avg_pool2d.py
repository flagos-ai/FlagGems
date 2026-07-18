import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_MAX_BLOCK_SIZE = 1024
_OUTPUT_TILE = 8


@libentry()
@triton.jit
def _adaptive_avg_pool2d_tiled_kernel(
    inp,
    out,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    MAX_WINDOW_H: tl.constexpr,
    MAX_WINDOW_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    width_tiles = tl.cdiv(out_w, BLOCK_W)
    tile_y = tile_idx // width_tiles
    tile_x = tile_idx % width_tiles
    out_y = tile_y * BLOCK_H + tl.arange(0, BLOCK_H)
    out_x = tile_x * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (out_y[:, None] < out_h) & (out_x[None, :] < out_w)

    in_y_start = out_y[:, None] * in_h // out_h
    in_y_end = ((out_y[:, None] + 1) * in_h + out_h - 1) // out_h
    in_x_start = out_x[None, :] * in_w // out_w
    in_x_end = ((out_x[None, :] + 1) * in_w + out_w - 1) // out_w
    window_area = (in_y_end - in_y_start) * (in_x_end - in_x_start)

    channel = nc_idx % in_c
    batch = nc_idx // in_c
    base = inp + batch * in_stride_n + channel * in_stride_c
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for linear in tl.range(0, MAX_WINDOW_H * MAX_WINDOW_W):
        offset_y = linear // MAX_WINDOW_W
        offset_x = linear % MAX_WINDOW_W
        in_y = in_y_start + offset_y
        y_mask = in_y < in_y_end
        in_x = in_x_start + offset_x
        mask = out_mask & y_mask & (in_x < in_x_end)
        values = tl.load(
            base + in_y * in_stride_h + in_x * in_stride_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += values

    out_offsets = nc_idx * out_h * out_w + out_y[:, None] * out_w + out_x[None, :]
    tl.store(out + out_offsets, acc / window_area, mask=out_mask)


@libentry()
@triton.jit
def _adaptive_avg_pool2d_kernel(
    inp,
    out,
    in_c,
    in_h,
    in_w,
    out_h,
    out_w,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    MAX_WINDOW_W: tl.constexpr,
    MAX_WINDOW_AREA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    out_idx = tl.program_id(0).to(tl.int64)
    out_x = out_idx % out_w
    out_y = (out_idx // out_w) % out_h
    nc_idx = out_idx // (out_h * out_w)
    channel = nc_idx % in_c
    batch = nc_idx // in_c

    in_y_start = out_y * in_h // out_h
    in_y_end = ((out_y + 1) * in_h + out_h - 1) // out_h
    in_x_start = out_x * in_w // out_w
    in_x_end = ((out_x + 1) * in_w + out_w - 1) // out_w
    window_h = in_y_end - in_y_start
    window_w = in_x_end - in_x_start

    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.full((), 0.0, dtype=tl.float32)
    base = inp + batch * in_stride_n + channel * in_stride_c
    for start in tl.static_range(0, MAX_WINDOW_AREA, BLOCK_SIZE):
        linear = start + offsets
        offset_y = linear // MAX_WINDOW_W
        offset_x = linear % MAX_WINDOW_W
        mask = (offset_y < window_h) & (offset_x < window_w)
        values = tl.load(
            base
            + (in_y_start + offset_y) * in_stride_h
            + (in_x_start + offset_x) * in_stride_w,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(values, axis=0)

    tl.store(out + out_idx, acc / (window_h * window_w).to(tl.float32))


def _max_window_size(input_size, output_size):
    return max(
        ((out_idx + 1) * input_size + output_size - 1) // output_size
        - out_idx * input_size // output_size
        for out_idx in range(output_size)
    )


def adaptive_avg_pool2d(inp: torch.Tensor, output_size):
    logger.debug("GEMS_MTHREADS ADAPTIVE_AVG_POOL2D")

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out_h, out_w = output_size

    unbatched = inp.ndim == 3
    if unbatched:
        inp = inp.unsqueeze(0)
    inp = inp.contiguous()
    in_n, in_c, in_h, in_w = inp.shape

    out = torch.empty((in_n, in_c, out_h, out_w), device=inp.device, dtype=inp.dtype)
    if out.numel() == 0 or in_h == 0 or in_w == 0:
        return out.squeeze(0) if unbatched else out

    max_window_h = _max_window_size(in_h, out_h)
    max_window_w = _max_window_size(in_w, out_w)
    max_window_area = max_window_h * max_window_w
    block_size = min(triton.next_power_of_2(max_window_area), _MAX_BLOCK_SIZE)
    num_warps = 1 if block_size <= 128 else 4

    with torch_device_fn.device(inp.device):
        if max_window_area <= 4096 and out_h * out_w >= 16:
            grid = (
                in_n * in_c,
                triton.cdiv(out_h, _OUTPUT_TILE) * triton.cdiv(out_w, _OUTPUT_TILE),
            )
            _adaptive_avg_pool2d_tiled_kernel[grid](
                inp,
                out,
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                *inp.stride(),
                max_window_h,
                max_window_w,
                _OUTPUT_TILE,
                _OUTPUT_TILE,
                num_warps=4,
                num_stages=1,
            )
        else:
            _adaptive_avg_pool2d_kernel[(out.numel(),)](
                inp,
                out,
                in_c,
                in_h,
                in_w,
                out_h,
                out_w,
                *inp.stride(),
                max_window_w,
                max_window_area,
                block_size,
                num_warps=num_warps,
                num_stages=1,
            )

    return out.squeeze(0) if unbatched else out
