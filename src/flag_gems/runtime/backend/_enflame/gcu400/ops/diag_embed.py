# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``diag_embed``：按输出规模分档的两条 Triton 路径。

公共实现先用 ``torch.zeros`` 清零整块输出，再用 ``pointwise_dynamic``
生成的 kernel 向 ``torch.diagonal`` 得到的非连续 view 写对角线；后者在
GCU 上会被多维索引分解拖慢，小 shape 还要多付一次 launch。

* 小输出：``torch.empty`` + 单个一维 flat kernel 一次写满输出，对角线取
  输入值、其余写 0，省掉清零 launch 与 codegen 多维分解。
* 大输出：保留 ``torch.zeros`` 的高效整块清零，只把对角拷贝换成按输入
  平坦索引连续读、按对角 stride 散写的一维 kernel，避免逐元素分解整个
  输出（大输出上整数除法代价远超收益）。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代按轮次生成；均为编译期/启动期配置，不改变算子语义。
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 1
FALLBACK_TO_ORIGINAL = 0
# 输出元素数超过该阈值时走 zeros + 对角散写路径：
# 单 kernel 的逐元素坐标分解在大输出上会被整数除法拖垮。
SMALL_OUTPUT_LIMIT = 2097152
# 大输出路径是否用手写散写 kernel 替代 pointwise_dynamic codegen。
LARGE_USE_SCATTER = 1
SCATTER_BLOCK_SIZE = 8192
SCATTER_NUM_WARPS = 4
# scatter kernel 用 int64 地址，可覆盖超过 2^30 元素的输出；保留上限
# 作为极端规模的保护，正常 case 不会触发。
SCATTER_INDEX_LIMIT = 1 << 40
# --- TUNING PARAMS END ---

# kernel 按固定 4 维展开坐标分解；实际 rank 不足 4 时多余维填 size=1。
_MAX_RANK = 4


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _diagonal_copy_func(inp):
    """原始路径的对角 view 拷贝，模块级只创建一次以便复用编译缓存。"""
    return inp


@libentry()
@triton.jit(do_not_specialize=["TOTAL"])
def _diag_embed_fill_kernel(
    x_ptr,
    y_ptr,
    TOTAL,
    Y_STRIDE_0: tl.constexpr,
    Y_STRIDE_1: tl.constexpr,
    Y_STRIDE_2: tl.constexpr,
    Y_STRIDE_3: tl.constexpr,
    Y_SIZE_0: tl.constexpr,
    Y_SIZE_1: tl.constexpr,
    Y_SIZE_2: tl.constexpr,
    Y_SIZE_3: tl.constexpr,
    X_COEFF_0: tl.constexpr,
    X_COEFF_1: tl.constexpr,
    X_COEFF_2: tl.constexpr,
    X_COEFF_3: tl.constexpr,
    DIM1: tl.constexpr,
    DIM2: tl.constexpr,
    OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """固定 grid + 核内跨步，把对角线与零值一次写满输出。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(TOTAL, BLOCK), num_programs):
        output_index = block_id * BLOCK + arange
        valid = output_index < TOTAL
        # 输出是连续 tensor，逐维反解坐标即可得到两个对角轴的位置。
        coord_0 = (output_index // Y_STRIDE_0) % Y_SIZE_0
        coord_1 = (output_index // Y_STRIDE_1) % Y_SIZE_1
        coord_2 = (output_index // Y_STRIDE_2) % Y_SIZE_2
        coord_3 = (output_index // Y_STRIDE_3) % Y_SIZE_3
        if DIM1 == 0:
            diag_coord_1 = coord_0
        elif DIM1 == 1:
            diag_coord_1 = coord_1
        elif DIM1 == 2:
            diag_coord_1 = coord_2
        else:
            diag_coord_1 = coord_3
        if DIM2 == 0:
            diag_coord_2 = coord_0
        elif DIM2 == 1:
            diag_coord_2 = coord_1
        elif DIM2 == 2:
            diag_coord_2 = coord_2
        else:
            diag_coord_2 = coord_3
        # offset>0 时对角线序号取 dim1 坐标，offset<0 时取 dim2 坐标，
        # 两者统一为较小值；两轴坐标差恰为 offset 才是有效对角线元素。
        on_diagonal = (diag_coord_2 - diag_coord_1 == OFFSET) & valid
        diagonal_index = tl.minimum(diag_coord_1, diag_coord_2)
        input_index = (
            coord_0 * X_COEFF_0
            + coord_1 * X_COEFF_1
            + coord_2 * X_COEFF_2
            + coord_3 * X_COEFF_3
            + diagonal_index
        )
        value = tl.load(x_ptr + input_index, mask=on_diagonal, other=0)
        tl.store(y_ptr + output_index, value, mask=valid)


@libentry()
@triton.jit(do_not_specialize=["TOTAL"])
def _diag_embed_scatter_kernel(
    x_ptr,
    y_ptr,
    TOTAL,
    N: tl.constexpr,
    DIAG_STRIDE: tl.constexpr,
    OFFSET_TERM: tl.constexpr,
    OUTER_SIZE_0: tl.constexpr,
    OUTER_SIZE_1: tl.constexpr,
    OUTER_Y_STRIDE_0: tl.constexpr,
    OUTER_Y_STRIDE_1: tl.constexpr,
    NUM_OUTER_AXES: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """按输入平坦索引连续读，按对角 stride 散写到已清零的输出。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(TOTAL, BLOCK), num_programs):
        input_index = block_id * BLOCK + arange
        valid = input_index < TOTAL
        # x 连续存放为 [outer..., N]，最后一维即对角线序号。
        diagonal_index = input_index % N
        outer_index = input_index // N
        if NUM_OUTER_AXES == 1:
            outer_coord_0 = outer_index % OUTER_SIZE_0
            outer_coord_1 = 0
        else:
            outer_coord_0 = outer_index // OUTER_SIZE_1
            outer_coord_1 = outer_index % OUTER_SIZE_1
        # 对角线每前进一格，dim1 与 dim2 坐标同时 +1；offset 只贡献常数项。
        # 输出可超过 2^30 个元素，地址必须用 int64，否则 GCU 会截断高位。
        output_index = (
            outer_coord_0.to(tl.int64) * OUTER_Y_STRIDE_0
            + outer_coord_1.to(tl.int64) * OUTER_Y_STRIDE_1
            + diagonal_index.to(tl.int64) * DIAG_STRIDE
            + OFFSET_TERM
        )
        value = tl.load(x_ptr + input_index, mask=valid)
        tl.store(y_ptr + output_index, value, mask=valid)


def _diag_embed_original(x, offset, dim1, dim2, y_shape):
    """原始 zeros + diagonal-view 路径，用于超出平坦索引上限或强制对照。"""
    y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
    diagonal_view = torch.diagonal(y, offset, dim1, dim2)
    _diagonal_copy_func.instantiate(x.ndim)(x, out0=diagonal_view)
    return y


def _diag_embed_large(x, offset, dim1, dim2, y_shape, diagonal_length, y_strides):
    """大输出：zeros 整块清零 + 一维散写 kernel 写对角线。"""
    if not LARGE_USE_SCATTER:
        y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
        diagonal_view = torch.diagonal(y, offset, dim1, dim2)
        _diagonal_copy_func.instantiate(x.ndim)(x, out0=diagonal_view)
        return y

    if not x.is_contiguous():
        x = x.contiguous()

    rank = len(y_shape)
    outer_axes = [axis for axis in range(rank) if axis not in (dim1, dim2)]
    outer_sizes = [y_shape[axis] for axis in outer_axes]
    # 输入 outer 维按连续布局展开；最多两个 outer 轴（rank ≤ 4）。
    outer_size_0 = outer_sizes[0] if len(outer_sizes) >= 1 else 1
    outer_size_1 = outer_sizes[1] if len(outer_sizes) >= 2 else 1
    outer_y_stride_0 = y_strides[outer_axes[0]] if len(outer_axes) >= 1 else 0
    outer_y_stride_1 = y_strides[outer_axes[1]] if len(outer_axes) >= 2 else 0

    diag_stride = y_strides[dim1] + y_strides[dim2]
    # offset>0 时 dim2 坐标多 offset，offset<0 时 dim1 坐标多 -offset。
    offset_term = max(offset, 0) * y_strides[dim2] + max(-offset, 0) * y_strides[dim1]

    y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
    total_input_elements = x.numel()
    grid = (min(triton.cdiv(total_input_elements, SCATTER_BLOCK_SIZE), FIXED_GRID),)
    device_context = (
        torch_device_fn.device(x.device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        _diag_embed_scatter_kernel[grid](
            x,
            y,
            total_input_elements,
            N=diagonal_length,
            DIAG_STRIDE=diag_stride,
            OFFSET_TERM=offset_term,
            OUTER_SIZE_0=outer_size_0,
            OUTER_SIZE_1=outer_size_1,
            OUTER_Y_STRIDE_0=outer_y_stride_0,
            OUTER_Y_STRIDE_1=outer_y_stride_1,
            NUM_OUTER_AXES=len(outer_axes),
            BLOCK=SCATTER_BLOCK_SIZE,
            GRID=FIXED_GRID,
            num_warps=SCATTER_NUM_WARPS,
        )
    return y


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    logger.debug("GEMS_GCU400 DIAG_EMBED")

    rank = x.ndim + 1
    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"
    dim1 = dim1 % rank
    dim2 = dim2 % rank
    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    # 交换两个对角轴等价于取 offset 的相反数，先归一化到 dim1 < dim2。
    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    diagonal_length = x.size(-1)
    last_dim = diagonal_length + abs(offset)
    y_shape = list(x.shape)
    y_shape.pop()
    y_shape.insert(dim1, last_dim)
    y_shape.insert(dim2, last_dim)

    total_elements = 1
    for dimension in y_shape:
        total_elements *= dimension

    # y 按连续布局存放，逐维 stride 由 shape 后缀积得到。
    y_strides = [1] * rank
    for axis in range(rank - 2, -1, -1):
        y_strides[axis] = y_strides[axis + 1] * y_shape[axis + 1]

    if FALLBACK_TO_ORIGINAL or rank > _MAX_RANK:
        return _diag_embed_original(x, offset, dim1, dim2, y_shape)
    if total_elements > SMALL_OUTPUT_LIMIT:
        if total_elements > SCATTER_INDEX_LIMIT:
            return _diag_embed_original(x, offset, dim1, dim2, y_shape)
        return _diag_embed_large(
            x, offset, dim1, dim2, y_shape, diagonal_length, y_strides
        )

    if not x.is_contiguous():
        x = x.contiguous()

    # x 的维序是「y 中除两个对角轴外的维」再接上最后一维，
    # 每个 y 轴坐标对 x 平坦索引的系数就是其后所有 x 维长度的乘积。
    outer_axes = [axis for axis in range(rank) if axis not in (dim1, dim2)]
    outer_sizes = [y_shape[axis] for axis in outer_axes] + [diagonal_length]
    x_coeffs = [0] * rank
    for position, axis in enumerate(outer_axes):
        coefficient = 1
        for size in outer_sizes[position + 1 :]:
            coefficient *= size
        x_coeffs[axis] = coefficient

    pad_count = _MAX_RANK - rank
    y_strides += [1] * pad_count
    y_sizes = list(y_shape) + [1] * pad_count
    x_coeffs += [0] * pad_count

    y = torch.empty(y_shape, dtype=x.dtype, device=x.device)
    grid = (min(triton.cdiv(total_elements, BLOCK_SIZE), FIXED_GRID),)
    device_context = (
        torch_device_fn.device(x.device) if USE_DEVICE_CONTEXT else nullcontext()
    )
    with device_context:
        _diag_embed_fill_kernel[grid](
            x,
            y,
            total_elements,
            Y_STRIDE_0=y_strides[0],
            Y_STRIDE_1=y_strides[1],
            Y_STRIDE_2=y_strides[2],
            Y_STRIDE_3=y_strides[3],
            Y_SIZE_0=y_sizes[0],
            Y_SIZE_1=y_sizes[1],
            Y_SIZE_2=y_sizes[2],
            Y_SIZE_3=y_sizes[3],
            X_COEFF_0=x_coeffs[0],
            X_COEFF_1=x_coeffs[1],
            X_COEFF_2=x_coeffs[2],
            X_COEFF_3=x_coeffs[3],
            DIM1=dim1,
            DIM2=dim2,
            OFFSET=offset,
            BLOCK=BLOCK_SIZE,
            GRID=FIXED_GRID,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return y
