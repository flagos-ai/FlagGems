import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry
from flag_gems.utils.shape_utils import heuristics_for_tile_size

logger = logging.getLogger(__name__)

MAX_NDIM = 8


@triton.jit
def _decompose_and_flip(
    task_id,
    inp_ptr,
    out_ptr,
    shape0: tl.constexpr,
    shape1: tl.constexpr,
    shape2: tl.constexpr,
    shape3: tl.constexpr,
    shape4: tl.constexpr,
    shape5: tl.constexpr,
    shape6: tl.constexpr,
    shape7: tl.constexpr,
    inp_stride0: tl.constexpr,
    inp_stride1: tl.constexpr,
    inp_stride2: tl.constexpr,
    inp_stride3: tl.constexpr,
    inp_stride4: tl.constexpr,
    inp_stride5: tl.constexpr,
    inp_stride6: tl.constexpr,
    inp_stride7: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    out_stride2: tl.constexpr,
    out_stride3: tl.constexpr,
    out_stride4: tl.constexpr,
    out_stride5: tl.constexpr,
    out_stride6: tl.constexpr,
    out_stride7: tl.constexpr,
    flip0: tl.constexpr,
    flip1: tl.constexpr,
    flip2: tl.constexpr,
    flip3: tl.constexpr,
    flip4: tl.constexpr,
    flip5: tl.constexpr,
    flip6: tl.constexpr,
    flip7: tl.constexpr,
    num_tasks,
    ndim: tl.constexpr,
):
    mask = task_id < num_tasks

    # Decompose linear index into multi-dimensional indices using output strides.
    # Output is contiguous (empty_like), so we use out_strides for index decomposition
    # and inp_strides for reading from the (possibly non-contiguous) input.
    remaining = task_id
    inp_offset = tl.zeros_like(task_id)
    out_offset = tl.zeros_like(task_id)

    if ndim > 0:
        idx = remaining // out_stride0 if out_stride0 > 0 else remaining
        if ndim > 1:
            remaining = remaining % out_stride0 if out_stride0 > 0 else remaining
        flipped_idx = (shape0 - 1 - idx) if flip0 else idx
        inp_offset += flipped_idx * inp_stride0
        out_offset += idx * out_stride0
    if ndim > 1:
        idx = remaining // out_stride1 if out_stride1 > 0 else remaining
        if ndim > 2:
            remaining = remaining % out_stride1 if out_stride1 > 0 else remaining
        flipped_idx = (shape1 - 1 - idx) if flip1 else idx
        inp_offset += flipped_idx * inp_stride1
        out_offset += idx * out_stride1
    if ndim > 2:
        idx = remaining // out_stride2 if out_stride2 > 0 else remaining
        if ndim > 3:
            remaining = remaining % out_stride2 if out_stride2 > 0 else remaining
        flipped_idx = (shape2 - 1 - idx) if flip2 else idx
        inp_offset += flipped_idx * inp_stride2
        out_offset += idx * out_stride2
    if ndim > 3:
        idx = remaining // out_stride3 if out_stride3 > 0 else remaining
        if ndim > 4:
            remaining = remaining % out_stride3 if out_stride3 > 0 else remaining
        flipped_idx = (shape3 - 1 - idx) if flip3 else idx
        inp_offset += flipped_idx * inp_stride3
        out_offset += idx * out_stride3
    if ndim > 4:
        idx = remaining // out_stride4 if out_stride4 > 0 else remaining
        if ndim > 5:
            remaining = remaining % out_stride4 if out_stride4 > 0 else remaining
        flipped_idx = (shape4 - 1 - idx) if flip4 else idx
        inp_offset += flipped_idx * inp_stride4
        out_offset += idx * out_stride4
    if ndim > 5:
        idx = remaining // out_stride5 if out_stride5 > 0 else remaining
        if ndim > 6:
            remaining = remaining % out_stride5 if out_stride5 > 0 else remaining
        flipped_idx = (shape5 - 1 - idx) if flip5 else idx
        inp_offset += flipped_idx * inp_stride5
        out_offset += idx * out_stride5
    if ndim > 6:
        idx = remaining // out_stride6 if out_stride6 > 0 else remaining
        if ndim > 7:
            remaining = remaining % out_stride6 if out_stride6 > 0 else remaining
        flipped_idx = (shape6 - 1 - idx) if flip6 else idx
        inp_offset += flipped_idx * inp_stride6
        out_offset += idx * out_stride6
    if ndim > 7:
        idx = remaining
        flipped_idx = (shape7 - 1 - idx) if flip7 else idx
        inp_offset += flipped_idx * inp_stride7
        out_offset += idx * out_stride7

    val = tl.load(inp_ptr + inp_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)


@libentry()
@triton.jit
def flip_kernel(
    inp_ptr,
    out_ptr,
    shape0: tl.constexpr,
    shape1: tl.constexpr,
    shape2: tl.constexpr,
    shape3: tl.constexpr,
    shape4: tl.constexpr,
    shape5: tl.constexpr,
    shape6: tl.constexpr,
    shape7: tl.constexpr,
    inp_stride0: tl.constexpr,
    inp_stride1: tl.constexpr,
    inp_stride2: tl.constexpr,
    inp_stride3: tl.constexpr,
    inp_stride4: tl.constexpr,
    inp_stride5: tl.constexpr,
    inp_stride6: tl.constexpr,
    inp_stride7: tl.constexpr,
    out_stride0: tl.constexpr,
    out_stride1: tl.constexpr,
    out_stride2: tl.constexpr,
    out_stride3: tl.constexpr,
    out_stride4: tl.constexpr,
    out_stride5: tl.constexpr,
    out_stride6: tl.constexpr,
    out_stride7: tl.constexpr,
    flip0: tl.constexpr,
    flip1: tl.constexpr,
    flip2: tl.constexpr,
    flip3: tl.constexpr,
    flip4: tl.constexpr,
    flip5: tl.constexpr,
    flip6: tl.constexpr,
    flip7: tl.constexpr,
    num_tasks,
    ndim: tl.constexpr,
    tiles_per_cta: int,
    tile_size: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    if one_tile_per_cta:
        task_id = pid * tile_size + tl.arange(0, tile_size)
        _decompose_and_flip(
            task_id,
            inp_ptr,
            out_ptr,
            shape0,
            shape1,
            shape2,
            shape3,
            shape4,
            shape5,
            shape6,
            shape7,
            inp_stride0,
            inp_stride1,
            inp_stride2,
            inp_stride3,
            inp_stride4,
            inp_stride5,
            inp_stride6,
            inp_stride7,
            out_stride0,
            out_stride1,
            out_stride2,
            out_stride3,
            out_stride4,
            out_stride5,
            out_stride6,
            out_stride7,
            flip0,
            flip1,
            flip2,
            flip3,
            flip4,
            flip5,
            flip6,
            flip7,
            num_tasks,
            ndim,
        )
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            task_id = tile_id * tile_size + tl.arange(0, tile_size)
            _decompose_and_flip(
                task_id,
                inp_ptr,
                out_ptr,
                shape0,
                shape1,
                shape2,
                shape3,
                shape4,
                shape5,
                shape6,
                shape7,
                inp_stride0,
                inp_stride1,
                inp_stride2,
                inp_stride3,
                inp_stride4,
                inp_stride5,
                inp_stride6,
                inp_stride7,
                out_stride0,
                out_stride1,
                out_stride2,
                out_stride3,
                out_stride4,
                out_stride5,
                out_stride6,
                out_stride7,
                flip0,
                flip1,
                flip2,
                flip3,
                flip4,
                flip5,
                flip6,
                flip7,
                num_tasks,
                ndim,
            )


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS ILUVATAR FLIP")
    ndim = A.dim()
    flip_dims_b = [False] * max(ndim, 1)
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        assert not flip_dims_b[
            dim
        ], "dim {} appears multiple times in the list of dims".format(dim)
        flip_dims_b[dim] = True

    n = sum(
        1 for i in range(ndim) if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0
    )
    if n == 0 or A.numel() <= 1:
        return A.clone()

    out = torch.empty_like(A)
    num_tasks = A.numel()

    # Pad shape/strides/flip to MAX_NDIM
    shape_padded = list(A.shape) + [1] * (MAX_NDIM - ndim)
    inp_strides = list(A.stride()) + [0] * (MAX_NDIM - ndim)
    out_strides = list(out.stride()) + [0] * (MAX_NDIM - ndim)
    flip_padded = [flip_dims_b[i] if i < ndim else False for i in range(MAX_NDIM)]

    tile_sizes = heuristics_for_tile_size(1024, num_tasks)
    tile_size = tile_sizes[0]

    num_tiles = triton.cdiv(num_tasks, tile_size)
    num_ctas = min(65536, num_tiles)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    one_tile_per_cta = tiles_per_cta == 1

    if tile_size < 2048:
        num_warps = 4
    elif tile_size < 4096:
        num_warps = 8
    else:
        num_warps = 16

    grid = (num_ctas, 1, 1)

    with torch_device_fn.device(A.device.index):
        flip_kernel[grid](
            A,
            out,
            shape_padded[0],
            shape_padded[1],
            shape_padded[2],
            shape_padded[3],
            shape_padded[4],
            shape_padded[5],
            shape_padded[6],
            shape_padded[7],
            inp_strides[0],
            inp_strides[1],
            inp_strides[2],
            inp_strides[3],
            inp_strides[4],
            inp_strides[5],
            inp_strides[6],
            inp_strides[7],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            out_strides[3],
            out_strides[4],
            out_strides[5],
            out_strides[6],
            out_strides[7],
            flip_padded[0],
            flip_padded[1],
            flip_padded[2],
            flip_padded[3],
            flip_padded[4],
            flip_padded[5],
            flip_padded[6],
            flip_padded[7],
            num_tasks,
            ndim,
            tiles_per_cta=tiles_per_cta,
            tile_size=tile_size,
            one_tile_per_cta=one_tile_per_cta,
            num_warps=num_warps,
        )

    return out
