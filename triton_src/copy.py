import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import (
    heuristics_for_num_warps,
    heuristics_for_tile_size,
    stride_order,
)


@triton.jit
def _copy_kernel(src):
    return src


def copy_kernel_wrapper_rank_0(in0: torch.Tensor, /, *, out0: torch.Tensor):
    """Host-side wrapper mirroring pointwise_dynamic rank-0 codegen."""
    assert in0.shape == out0.shape, "operand shapes mismatch"
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    num_warps = 1
    num_ctas = 1
    grid = (num_ctas, 1, 1)
    with torch_device_fn.device(in0.device.index):
        _copy_kernel_cppwrapper_rank_0[grid](in0, out0, num_warps=num_warps)
    return out0


def copy_kernel_wrapper_rank_1(in0: torch.Tensor, /, *, out0: torch.Tensor):
    """Host-side wrapper mirroring pointwise_dynamic rank-1 codegen."""
    assert in0.shape == out0.shape, "operand shapes mismatch"
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(512, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)
    )
    num_ctas = num_tiles
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)
    in0_strides = in0.stride()
    in0_stride_order = stride_order(in0_strides)
    out0_strides = out0.stride()
    out0_stride_order = stride_order(out0_strides)
    with torch_device_fn.device(in0.device.index):
        _copy_kernel_cppwrapper_rank_1[grid](
            in0,
            out0,
            in0_strides[0],
            in0_stride_order[0],
            out0_strides[0],
            out0_stride_order[0],
            shape[0],
            num_tasks,
            tiles_per_cta,
            tile_sizes[0],
            one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


def copy_kernel_wrapper_rank_2(in0: torch.Tensor, /, *, out0: torch.Tensor):
    """Host-side wrapper mirroring pointwise_dynamic rank-2 codegen."""
    assert in0.shape == out0.shape, "operand shapes mismatch"
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(512, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)
    )
    num_ctas = num_tiles
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)
    in0_strides = in0.stride()
    in0_stride_order = stride_order(in0_strides)
    out0_strides = out0.stride()
    out0_stride_order = stride_order(out0_strides)
    with torch_device_fn.device(in0.device.index):
        _copy_kernel_cppwrapper_rank_2[grid](
            in0,
            out0,
            in0_strides[0],
            in0_strides[1],
            in0_stride_order[0],
            in0_stride_order[1],
            out0_strides[0],
            out0_strides[1],
            out0_stride_order[0],
            out0_stride_order[1],
            shape[0],
            shape[1],
            num_tasks,
            tiles_per_cta,
            tile_sizes[0],
            tile_sizes[1],
            one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


def copy_kernel_wrapper_rank_3(in0: torch.Tensor, /, *, out0: torch.Tensor):
    """Host-side wrapper mirroring pointwise_dynamic rank-3 codegen."""
    assert in0.shape == out0.shape, "operand shapes mismatch"
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(512, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)
    )
    num_ctas = num_tiles
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)
    in0_strides = in0.stride()
    in0_stride_order = stride_order(in0_strides)
    out0_strides = out0.stride()
    out0_stride_order = stride_order(out0_strides)
    with torch_device_fn.device(in0.device.index):
        _copy_kernel_cppwrapper_rank_3[grid](
            in0,
            out0,
            in0_strides[0],
            in0_strides[1],
            in0_strides[2],
            in0_stride_order[0],
            in0_stride_order[1],
            in0_stride_order[2],
            out0_strides[0],
            out0_strides[1],
            out0_strides[2],
            out0_stride_order[0],
            out0_stride_order[1],
            out0_stride_order[2],
            shape[0],
            shape[1],
            shape[2],
            num_tasks,
            tiles_per_cta,
            tile_sizes[0],
            tile_sizes[1],
            tile_sizes[2],
            one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


def copy_kernel_wrapper_rank_4(in0: torch.Tensor, /, *, out0: torch.Tensor):
    """Host-side wrapper mirroring pointwise_dynamic rank-4 codegen."""
    assert in0.shape == out0.shape, "operand shapes mismatch"
    shape = out0.shape
    num_tasks = out0.numel()
    if num_tasks == 0:
        return out0
    tile_sizes = heuristics_for_tile_size(512, *shape)
    tile_size = math.prod(tile_sizes)
    num_tiles = math.prod(
        triton.cdiv(size, tile_size) for size, tile_size in zip(shape, tile_sizes)
    )
    num_ctas = num_tiles
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    num_warps = heuristics_for_num_warps(tile_size)
    one_tile_per_cta = tiles_per_cta == 1
    grid = (num_ctas, 1, 1)
    in0_strides = in0.stride()
    in0_stride_order = stride_order(in0_strides)
    out0_strides = out0.stride()
    out0_stride_order = stride_order(out0_strides)
    with torch_device_fn.device(in0.device.index):
        _copy_kernel_cppwrapper_rank_4[grid](
            in0,
            out0,
            in0_strides[0],
            in0_strides[1],
            in0_strides[2],
            in0_strides[3],
            in0_stride_order[0],
            in0_stride_order[1],
            in0_stride_order[2],
            in0_stride_order[3],
            out0_strides[0],
            out0_strides[1],
            out0_strides[2],
            out0_strides[3],
            out0_stride_order[0],
            out0_stride_order[1],
            out0_stride_order[2],
            out0_stride_order[3],
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            num_tasks,
            tiles_per_cta,
            tile_sizes[0],
            tile_sizes[1],
            tile_sizes[2],
            tile_sizes[3],
            one_tile_per_cta,
            num_warps=num_warps,
        )
    return out0


@triton.jit
def _copy_kernel_cppwrapper_rank_0(
    in0_ptr: tl.tensor,  # of tl.pointer_type
    out0_ptr: tl.tensor,  # of tl.pointer_type
):
    # loads
    in0 = tl.load(in0_ptr).to(in0_ptr.type.element_ty)

    # compute
    out0 = _copy_kernel(in0)

    # stores
    tl.store(out0_ptr, out0.to(out0_ptr.type.element_ty))


@triton.jit
def _copy_kernel_cppwrapper_rank_1(
    in0_ptr: tl.tensor,  # of tl.pointer_type
    out0_ptr: tl.tensor,  # of tl.pointer_type
    in0_stride0: int,  # strides for in0
    in0_stride_order0: tl.constexpr,  # stride order for in0
    out0_stride0: int,  # strides for out0
    out0_stride_order0: tl.constexpr,  # stride order for out0
    s0,  # task_space
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    if one_tile_per_cta:  # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        # tile offsets
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        # loads
        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0,),
            (in0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(in0_stride_order0,),
        )
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
            in0_ptr.type.element_ty
        )  # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = _copy_kernel(in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0,),
            (out0_stride0,),
            (offset0,),
            (tile_size0,),
            order=(out0_stride_order0,),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(out0_stride_order0,),
        )
    else:  # grid-stride-loop style kernel
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            # tile offsets
            offset0 = (tile_id0 * tile_size0).to(tl.int32)
            # loads
            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0,),
                (in0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(in0_stride_order0,),
            )
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(
                in0_ptr.type.element_ty
            )  # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = _copy_kernel(in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0,),
                (out0_stride0,),
                (offset0,),
                (tile_size0,),
                order=(out0_stride_order0,),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(out0_stride_order0,),
            )


@triton.jit
def _copy_kernel_cppwrapper_rank_2(
    in0_ptr: tl.tensor,  # of tl.pointer_type
    out0_ptr: tl.tensor,  # of tl.pointer_type
    in0_stride0: int,
    in0_stride1: int,  # strides for in0
    in0_stride_order0: tl.constexpr,
    in0_stride_order1: tl.constexpr,  # stride order for in0
    out0_stride0: int,
    out0_stride1: int,  # strides for out0
    out0_stride_order0: tl.constexpr,
    out0_stride_order1: tl.constexpr,  # stride order for out0
    s0,
    s1,  # task_space
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    tile_size1: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    num_tiles1 = tl.cdiv(s1, tile_size1)
    if one_tile_per_cta:  # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id1 = tile_id % num_tiles1
        tile_id //= num_tiles1
        tile_id0 = tile_id

        # tile offsets
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        # loads
        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0, s1),
            (in0_stride0, in0_stride1),
            (offset0, offset1),
            (tile_size0, tile_size1),
            order=(in0_stride_order0, in0_stride_order1),
        )
        in0 = tl.load(
            in0_bptr, boundary_check=(in0_stride_order0, in0_stride_order1)
        ).to(
            in0_ptr.type.element_ty
        )  # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = _copy_kernel(in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0, s1),
            (out0_stride0, out0_stride1),
            (offset0, offset1),
            (tile_size0, tile_size1),
            order=(out0_stride_order0, out0_stride_order1),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(out0_stride_order0, out0_stride_order1),
        )
    else:  # grid-stride-loop style kernel
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id1 = tile_id % num_tiles1
            tile_id //= num_tiles1
            tile_id0 = tile_id

            # tile offsets
            offset0 = (tile_id0 * tile_size0).to(tl.int32)
            offset1 = (tile_id1 * tile_size1).to(tl.int32)
            # loads
            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0, s1),
                (in0_stride0, in0_stride1),
                (offset0, offset1),
                (tile_size0, tile_size1),
                order=(in0_stride_order0, in0_stride_order1),
            )
            in0 = tl.load(
                in0_bptr, boundary_check=(in0_stride_order0, in0_stride_order1)
            ).to(
                in0_ptr.type.element_ty
            )  # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = _copy_kernel(in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0, s1),
                (out0_stride0, out0_stride1),
                (offset0, offset1),
                (tile_size0, tile_size1),
                order=(out0_stride_order0, out0_stride_order1),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(out0_stride_order0, out0_stride_order1),
            )


@triton.jit
def _copy_kernel_cppwrapper_rank_3(
    in0_ptr: tl.tensor,  # of tl.pointer_type
    out0_ptr: tl.tensor,  # of tl.pointer_type
    in0_stride0: int,
    in0_stride1: int,
    in0_stride2: int,
    in0_stride_order0: tl.constexpr,
    in0_stride_order1: tl.constexpr,
    in0_stride_order2: tl.constexpr,
    out0_stride0: int,
    out0_stride1: int,
    out0_stride2: int,
    out0_stride_order0: tl.constexpr,
    out0_stride_order1: tl.constexpr,
    out0_stride_order2: tl.constexpr,
    s0,
    s1,
    s2,
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    tile_size1: tl.constexpr,
    tile_size2: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    num_tiles2 = tl.cdiv(s2, tile_size2)
    num_tiles1 = tl.cdiv(s1, tile_size1)
    if one_tile_per_cta:
        tile_id = pid
        tile_id2 = tile_id % num_tiles2
        tile_id //= num_tiles2
        tile_id1 = tile_id % num_tiles1
        tile_id //= num_tiles1
        tile_id0 = tile_id

        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        offset2 = (tile_id2 * tile_size2).to(tl.int32)
        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0, s1, s2),
            (in0_stride0, in0_stride1, in0_stride2),
            (offset0, offset1, offset2),
            (tile_size0, tile_size1, tile_size2),
            order=(
                in0_stride_order0,
                in0_stride_order1,
                in0_stride_order2,
            ),
        )
        in0 = tl.load(
            in0_bptr,
            boundary_check=(
                in0_stride_order0,
                in0_stride_order1,
                in0_stride_order2,
            ),
        ).to(in0_ptr.type.element_ty)

        out0 = _copy_kernel(in0)

        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0, s1, s2),
            (out0_stride0, out0_stride1, out0_stride2),
            (offset0, offset1, offset2),
            (tile_size0, tile_size1, tile_size2),
            order=(
                out0_stride_order0,
                out0_stride_order1,
                out0_stride_order2,
            ),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(
                out0_stride_order0,
                out0_stride_order1,
                out0_stride_order2,
            ),
        )
    else:
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            tile_id2 = tile_id % num_tiles2
            tile_id //= num_tiles2
            tile_id1 = tile_id % num_tiles1
            tile_id //= num_tiles1
            tile_id0 = tile_id

            offset0 = (tile_id0 * tile_size0).to(tl.int32)
            offset1 = (tile_id1 * tile_size1).to(tl.int32)
            offset2 = (tile_id2 * tile_size2).to(tl.int32)
            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0, s1, s2),
                (in0_stride0, in0_stride1, in0_stride2),
                (offset0, offset1, offset2),
                (tile_size0, tile_size1, tile_size2),
                order=(
                    in0_stride_order0,
                    in0_stride_order1,
                    in0_stride_order2,
                ),
            )
            in0 = tl.load(
                in0_bptr,
                boundary_check=(
                    in0_stride_order0,
                    in0_stride_order1,
                    in0_stride_order2,
                ),
            ).to(in0_ptr.type.element_ty)

            out0 = _copy_kernel(in0)

            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0, s1, s2),
                (out0_stride0, out0_stride1, out0_stride2),
                (offset0, offset1, offset2),
                (tile_size0, tile_size1, tile_size2),
                order=(
                    out0_stride_order0,
                    out0_stride_order1,
                    out0_stride_order2,
                ),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(
                    out0_stride_order0,
                    out0_stride_order1,
                    out0_stride_order2,
                ),
            )


@triton.jit
def _copy_kernel_cppwrapper_rank_4(
    in0_ptr: tl.tensor,  # of tl.pointer_type
    out0_ptr: tl.tensor,  # of tl.pointer_type
    in0_stride0: int,
    in0_stride1: int,
    in0_stride2: int,
    in0_stride3: int,
    in0_stride_order0: tl.constexpr,
    in0_stride_order1: tl.constexpr,
    in0_stride_order2: tl.constexpr,
    in0_stride_order3: tl.constexpr,
    out0_stride0: int,
    out0_stride1: int,
    out0_stride2: int,
    out0_stride3: int,
    out0_stride_order0: tl.constexpr,
    out0_stride_order1: tl.constexpr,
    out0_stride_order2: tl.constexpr,
    out0_stride_order3: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    num_tasks,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    tile_size1: tl.constexpr,
    tile_size2: tl.constexpr,
    tile_size3: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    num_tiles3 = tl.cdiv(s3, tile_size3)
    num_tiles2 = tl.cdiv(s2, tile_size2)
    num_tiles1 = tl.cdiv(s1, tile_size1)
    if one_tile_per_cta:
        tile_id = pid
        tile_id3 = tile_id % num_tiles3
        tile_id //= num_tiles3
        tile_id2 = tile_id % num_tiles2
        tile_id //= num_tiles2
        tile_id1 = tile_id % num_tiles1
        tile_id //= num_tiles1
        tile_id0 = tile_id

        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        offset2 = (tile_id2 * tile_size2).to(tl.int32)
        offset3 = (tile_id3 * tile_size3).to(tl.int32)
        in0_bptr = tl.make_block_ptr(
            in0_ptr,
            (s0, s1, s2, s3),
            (in0_stride0, in0_stride1, in0_stride2, in0_stride3),
            (offset0, offset1, offset2, offset3),
            (tile_size0, tile_size1, tile_size2, tile_size3),
            order=(
                in0_stride_order0,
                in0_stride_order1,
                in0_stride_order2,
                in0_stride_order3,
            ),
        )
        in0 = tl.load(
            in0_bptr,
            boundary_check=(
                in0_stride_order0,
                in0_stride_order1,
                in0_stride_order2,
                in0_stride_order3,
            ),
        ).to(in0_ptr.type.element_ty)

        out0 = _copy_kernel(in0)

        out0_bptr = tl.make_block_ptr(
            out0_ptr,
            (s0, s1, s2, s3),
            (out0_stride0, out0_stride1, out0_stride2, out0_stride3),
            (offset0, offset1, offset2, offset3),
            (tile_size0, tile_size1, tile_size2, tile_size3),
            order=(
                out0_stride_order0,
                out0_stride_order1,
                out0_stride_order2,
                out0_stride_order3,
            ),
        )
        tl.store(
            out0_bptr,
            out0.to(out0_bptr.type.element_ty),
            boundary_check=(
                out0_stride_order0,
                out0_stride_order1,
                out0_stride_order2,
                out0_stride_order3,
            ),
        )
    else:
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            tile_id3 = tile_id % num_tiles3
            tile_id //= num_tiles3
            tile_id2 = tile_id % num_tiles2
            tile_id //= num_tiles2
            tile_id1 = tile_id % num_tiles1
            tile_id //= num_tiles1
            tile_id0 = tile_id

            offset0 = (tile_id0 * tile_size0).to(tl.int32)
            offset1 = (tile_id1 * tile_size1).to(tl.int32)
            offset2 = (tile_id2 * tile_size2).to(tl.int32)
            offset3 = (tile_id3 * tile_size3).to(tl.int32)
            in0_bptr = tl.make_block_ptr(
                in0_ptr,
                (s0, s1, s2, s3),
                (in0_stride0, in0_stride1, in0_stride2, in0_stride3),
                (offset0, offset1, offset2, offset3),
                (tile_size0, tile_size1, tile_size2, tile_size3),
                order=(
                    in0_stride_order0,
                    in0_stride_order1,
                    in0_stride_order2,
                    in0_stride_order3,
                ),
            )
            in0 = tl.load(
                in0_bptr,
                boundary_check=(
                    in0_stride_order0,
                    in0_stride_order1,
                    in0_stride_order2,
                    in0_stride_order3,
                ),
            ).to(in0_ptr.type.element_ty)

            out0 = _copy_kernel(in0)

            out0_bptr = tl.make_block_ptr(
                out0_ptr,
                (s0, s1, s2, s3),
                (out0_stride0, out0_stride1, out0_stride2, out0_stride3),
                (offset0, offset1, offset2, offset3),
                (tile_size0, tile_size1, tile_size2, tile_size3),
                order=(
                    out0_stride_order0,
                    out0_stride_order1,
                    out0_stride_order2,
                    out0_stride_order3,
                ),
            )
            tl.store(
                out0_bptr,
                out0.to(out0_bptr.type.element_ty),
                boundary_check=(
                    out0_stride_order0,
                    out0_stride_order1,
                    out0_stride_order2,
                    out0_stride_order3,
                ),
            )


@triton.jit
def copy_kernel_linear(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    vals = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def copy_kernel_nd(
    src_ptr,
    dst_ptr,
    shape_ptr,
    src_stride_ptr,
    dst_stride_ptr,
    numel,
    NDIMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    linear = offs.to(tl.int64)

    src_offset = tl.zeros([BLOCK], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK], dtype=tl.int64)

    for d in range(NDIMS - 1, -1, -1):
        dim = tl.load(shape_ptr + d)
        idx = linear % dim
        linear = linear // dim
        src_stride = tl.load(src_stride_ptr + d)
        dst_stride = tl.load(dst_stride_ptr + d)
        src_offset += idx * src_stride
        dst_offset += idx * dst_stride

    val = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, val, mask=mask)
