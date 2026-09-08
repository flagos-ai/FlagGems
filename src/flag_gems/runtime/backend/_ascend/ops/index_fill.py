import importlib.util
import logging
import math
import os
import weakref
from typing import Any, Callable, Mapping, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from flag_gems.ops.index_fill import (
    _native_clone,
    _prepare_index,
    _prepare_tensor_value,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

_SMALL_INNER_BLOCK_I = 16
_SMALL_INNER_BLOCK_OUTER = 8
_SMALL_INNER_BLOCK_N = 4
# Avoid scatter-like small-inner updates once their two-dimensional grid is large.
_TRANSPOSE_FILL_MIN_SPARSE_PROGRAMS = 1600
_FULL_COVERAGE_HOST_CHECK_MAX_BYTES = 384 * 1024
_TRANSPOSE_FILL_SMALL_FULL_DIM_MAX_SIZE = 256
_TRANSPOSE_FILL_SMALL_FULL_DIM_MIN_NUMEL = 1024 * 1024


@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "outer_index_len",
        "index_len",
        "dim_size",
        "inner_size",
    ]
)
def index_fill_contiguous_scalar_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    inner_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    inner_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < outer_index_len
    index_coord = m_offsets % index_len
    outer_coord = m_offsets // index_len
    raw_index = tl.load(index + index_coord, mask=m_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert((~m_mask) | valid_index, "index out of bounds")
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index).to(
        tl.int64
    )

    out_offsets = outer_coord[:, None].to(tl.int64) * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]

    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask &= valid_index[:, None]
    tl.store(out + out_offsets, value, mask=store_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "outer_size",
        "index_len",
        "dim_size",
        "inner_size",
    ]
)
def index_fill_contiguous_scalar_small_inner_kernel(
    out,
    index,
    value,
    outer_size,
    index_len,
    dim_size,
    inner_size,
    HAS_NEGATIVE: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_OUTER: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_index = ext.program_id(axis=0)
    pid_outer = ext.program_id(axis=1)
    index_offsets = pid_index * BLOCK_I + tl.arange(0, BLOCK_I)
    index_mask = index_offsets < index_len
    inner_offsets = tl.arange(0, BLOCK_N)

    dim_size_i32 = dim_size.to(tl.int32)
    inner_size_i32 = inner_size.to(tl.int32)
    index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(tl.int32)
    if HAS_NEGATIVE:
        index_values = tl.where(
            index_values < 0, index_values + dim_size_i32, index_values
        )

    for outer_offset in range(0, BLOCK_OUTER):
        outer_index = pid_outer * BLOCK_OUTER + outer_offset
        outer_mask = outer_index < outer_size
        out_offsets = outer_index.to(tl.int32) * dim_size_i32 * inner_size_i32
        out_offsets += index_values[:, None] * inner_size_i32
        out_offsets += inner_offsets[None, :]
        store_mask = (
            outer_mask & index_mask[:, None] & (inner_offsets[None, :] < inner_size_i32)
        )
        tl.store(out + out_offsets, value, mask=store_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "outer_size",
        "dim_size",
        "inner_size",
    ]
)
def index_fill_contiguous_scalar_small_inner_blockptr_kernel(
    out,
    index,
    value,
    outer_size,
    dim_size,
    inner_size,
    HAS_NEGATIVE: tl.constexpr,
    BLOCK_OUTER: tl.constexpr,
    SPAN: tl.constexpr,
):
    pid_index = ext.program_id(axis=0)
    pid_outer = ext.program_id(axis=1)

    dim_size_i32 = dim_size.to(tl.int32)
    inner_size_i32 = inner_size.to(tl.int32)
    raw_index = tl.load(index + pid_index).to(tl.int32)
    valid_index = (raw_index >= -dim_size_i32) & (raw_index < dim_size_i32)
    index_value = raw_index
    if HAS_NEGATIVE:
        index_value = tl.where(index_value < 0, index_value + dim_size_i32, index_value)

    # Block pointers cannot accept a mask. Skip invalid indices and map the
    # final outer tail to row zero, which has already received the same value.
    if valid_index:
        for outer_offset in range(0, BLOCK_OUTER):
            outer_index = pid_outer * BLOCK_OUTER + outer_offset
            safe_outer_index = tl.where(outer_index < outer_size, outer_index, 0)
            base = (
                out
                + (safe_outer_index.to(tl.int32) * dim_size_i32 + index_value)
                * inner_size_i32
            )
            block = tl.make_block_ptr(
                base=base,
                shape=(SPAN,),
                strides=(1,),
                offsets=(0,),
                block_shape=(4,),
                order=(0,),
            )
            values = tl.full((4,), value, out.dtype.element_ty)
            tl.store(block, values, boundary_check=(0,))


@libentry()
@triton.jit(do_not_specialize=["N"])
def index_fill_contiguous_full_kernel(
    out,
    value,
    N,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    if VALUE_IS_TENSOR:
        fill_value = tl.load(value)
    else:
        fill_value = value
    tl.store(out + offsets, fill_value, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "outer_index_len",
        "index_len",
        "dim_size",
        "inner_size",
    ]
)
def index_fill_contiguous_tensor_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    inner_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    inner_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < outer_index_len
    index_coord = m_offsets % index_len
    outer_coord = m_offsets // index_len
    raw_index = tl.load(index + index_coord, mask=m_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert((~m_mask) | valid_index, "index out of bounds")
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index).to(
        tl.int64
    )

    out_offsets = outer_coord[:, None].to(tl.int64) * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]

    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask &= valid_index[:, None]
    fill_value = tl.load(value)
    tl.store(out + out_offsets, fill_value, mask=store_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "value",
        "index_len",
        "dim_size",
    ]
)
def index_fill_contiguous_scalar_inner1_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_outer = ext.program_id(axis=0)
    pid_index = ext.program_id(axis=1)
    index_offsets = pid_index.to(tl.int32) * BLOCK_I + tl.arange(0, BLOCK_I)
    index_mask = index_offsets < index_len.to(tl.int32)

    if USE_INT32:
        dim_size_i32 = dim_size.to(tl.int32)
        index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
            tl.int32
        )
        if HAS_NEGATIVE:
            index_values = tl.where(
                index_values < 0, index_values + dim_size_i32, index_values
            )
        out_offsets = pid_outer.to(tl.int32) * dim_size_i32 + index_values
    else:
        dim_size_i64 = dim_size.to(tl.int64)
        index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
            tl.int64
        )
        if HAS_NEGATIVE:
            index_values = tl.where(
                index_values < 0, index_values + dim_size_i64, index_values
            )
        out_offsets = pid_outer.to(tl.int64) * dim_size_i64 + index_values

    tl.store(out + out_offsets, value, mask=index_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "index_len",
        "dim_size",
    ]
)
def index_fill_contiguous_tensor_inner1_kernel(
    out,
    index,
    value,
    index_len,
    dim_size,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_outer = ext.program_id(axis=0)
    pid_index = ext.program_id(axis=1)
    index_offsets = pid_index.to(tl.int32) * BLOCK_I + tl.arange(0, BLOCK_I)
    index_mask = index_offsets < index_len.to(tl.int32)

    if USE_INT32:
        dim_size_i32 = dim_size.to(tl.int32)
        index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
            tl.int32
        )
        if HAS_NEGATIVE:
            index_values = tl.where(
                index_values < 0, index_values + dim_size_i32, index_values
            )
        out_offsets = pid_outer.to(tl.int32) * dim_size_i32 + index_values
    else:
        dim_size_i64 = dim_size.to(tl.int64)
        index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
            tl.int64
        )
        if HAS_NEGATIVE:
            index_values = tl.where(
                index_values < 0, index_values + dim_size_i64, index_values
            )
        out_offsets = pid_outer.to(tl.int64) * dim_size_i64 + index_values

    tl.store(out + out_offsets, tl.load(value), mask=index_mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "index_len",
        "dim_size",
    ]
)
def index_fill_membership_mask_kernel(
    membership,
    index,
    index_len,
    dim_size,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    worker_count = ext.num_programs(axis=0)

    for block_start in range(pid * BLOCK_I, index_len, worker_count * BLOCK_I):
        index_offsets = block_start + tl.arange(0, BLOCK_I)
        index_mask = index_offsets < index_len

        if USE_INT32:
            dim_size_i32 = dim_size.to(tl.int32)
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int32
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size_i32, index_values
                )
        else:
            dim_size_i64 = dim_size.to(tl.int64)
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int64
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size_i64, index_values
                )

        # Duplicated indices are legal. A nonzero value marks membership.
        tl.atomic_add(membership + index_values, 1, mask=index_mask)


@libentry()
@triton.jit(do_not_specialize=["index_len", "dim_size"])
def index_fill_membership_mask_build_kernel(
    membership,
    index,
    index_len,
    dim_size,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    for zero_start in range(0, dim_size, BLOCK_N):
        zero_offsets = zero_start + tl.arange(0, BLOCK_N)
        zero_mask = zero_offsets < dim_size
        tl.store(membership + zero_offsets, 0, mask=zero_mask)

    for index_start in range(0, index_len, BLOCK_I):
        index_offsets = index_start + tl.arange(0, BLOCK_I)
        index_mask = index_offsets < index_len
        if USE_INT32:
            dim_size_i32 = dim_size.to(tl.int32)
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int32
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size_i32, index_values
                )
        else:
            dim_size_i64 = dim_size.to(tl.int64)
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int64
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size_i64, index_values
                )

        # One program initializes and marks the buffer, preserving duplicate indices.
        tl.atomic_add(membership + index_values, 1, mask=index_mask)


@libentry()
@triton.jit(do_not_specialize=["index_len", "row_count", "row_width"])
def index_fill_contiguous_dim0_rows_kernel(
    out,
    index,
    value,
    index_len,
    row_count,
    row_width,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = ext.program_id(axis=0)
    row_mask = row_id < index_len

    if USE_INT32:
        row_count_value = row_count.to(tl.int32)
        row_width_value = row_width.to(tl.int32)
        index_value = tl.load(index + row_id, mask=row_mask, other=0).to(tl.int32)
        if HAS_NEGATIVE:
            index_value = tl.where(
                index_value < 0, index_value + row_count_value, index_value
            )
        row_offset = index_value * row_width_value
    else:
        row_count_value = row_count.to(tl.int64)
        row_width_value = row_width.to(tl.int64)
        index_value = tl.load(index + row_id, mask=row_mask, other=0).to(tl.int64)
        if HAS_NEGATIVE:
            index_value = tl.where(
                index_value < 0, index_value + row_count_value, index_value
            )
        row_offset = index_value * row_width_value

    if VALUE_IS_TENSOR:
        value_scalar = tl.load(value)
    else:
        value_scalar = value

    for column_start in range(0, row_width, BLOCK_N):
        columns = column_start + tl.arange(0, BLOCK_N)
        mask = row_mask & (columns < row_width)
        tl.store(out + row_offset + columns, value_scalar, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["outer_size", "dim_size"])
def index_fill_contiguous_mask_inner1_reuse_kernel(
    out,
    membership,
    value,
    outer_size,
    dim_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    outer_size_i32 = outer_size.to(tl.int32)
    dim_size_i32 = dim_size.to(tl.int32)
    n_tiles = tl.cdiv(dim_size_i32, BLOCK_N)
    n_tile = pid.to(tl.int32) % n_tiles
    outer_block = pid.to(tl.int32) // n_tiles
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < dim_size_i32
    selected = tl.load(membership + n_offsets, mask=n_mask, other=0) > 0

    if VALUE_IS_TENSOR:
        value_scalar = tl.load(value)
    else:
        value_scalar = value

    for row_offset in tl.range(BLOCK_P):
        outer_id = outer_block * BLOCK_P + row_offset
        row_mask = n_mask & (outer_id < outer_size_i32)
        out_offsets = outer_id * dim_size_i32 + n_offsets
        original = tl.load(out + out_offsets, mask=row_mask, other=0)
        fill_values = tl.full([BLOCK_N], value_scalar, dtype=original.dtype)
        result = tl.where(selected, fill_values, original)
        tl.store(out + out_offsets, result, mask=row_mask)


@libentry()
@triton.jit(do_not_specialize=["outer_size", "dim_size"])
def index_fill_contiguous_mask_inner1_copy_reuse_kernel(
    inp,
    out,
    membership,
    value,
    outer_size,
    dim_size,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    outer_size_i32 = outer_size.to(tl.int32)
    dim_size_i32 = dim_size.to(tl.int32)
    n_tiles = tl.cdiv(dim_size_i32, BLOCK_N)
    n_tile = pid.to(tl.int32) % n_tiles
    outer_block = pid.to(tl.int32) // n_tiles
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < dim_size_i32
    selected = tl.load(membership + n_offsets, mask=n_mask, other=0) > 0

    if VALUE_IS_TENSOR:
        value_scalar = tl.load(value)
    else:
        value_scalar = value

    for row_offset in tl.range(BLOCK_P):
        outer_id = outer_block * BLOCK_P + row_offset
        row_mask = n_mask & (outer_id < outer_size_i32)
        offsets = outer_id * dim_size_i32 + n_offsets
        original = tl.load(inp + offsets, mask=row_mask, other=0)
        fill_values = tl.full([BLOCK_N], value_scalar, dtype=original.dtype)
        result = tl.where(selected, fill_values, original)
        tl.store(out + offsets, result, mask=row_mask)


@libentry()
@triton.jit(do_not_specialize=["index_len", "outer_size", "dim_size"])
def index_fill_contiguous_local_membership_inner1_kernel(
    inp,
    out,
    index,
    value,
    index_len,
    outer_size,
    dim_size,
    HAS_NEGATIVE: tl.constexpr,
    USE_INT32: tl.constexpr,
    VALUE_IS_TENSOR: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = ext.program_id(axis=0)
    outer_size_i32 = outer_size.to(tl.int32)
    dim_size_i32 = dim_size.to(tl.int32)
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < dim_size_i32
    selected = tl.full([BLOCK_N], value=0, dtype=tl.int1)

    for index_start in range(0, index_len, BLOCK_I):
        index_offsets = index_start + tl.arange(0, BLOCK_I)
        index_mask = index_offsets < index_len
        if USE_INT32:
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int32
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size_i32, index_values
                )
        else:
            index_values = tl.load(index + index_offsets, mask=index_mask, other=0).to(
                tl.int64
            )
            if HAS_NEGATIVE:
                index_values = tl.where(
                    index_values < 0, index_values + dim_size, index_values
                )

        matches = (n_offsets[:, None] == index_values[None, :]) & index_mask[None, :]
        selected = selected | (tl.sum(matches.to(tl.int32), axis=1) > 0)

    if VALUE_IS_TENSOR:
        value_scalar = tl.load(value)
    else:
        value_scalar = value

    for row_offset in tl.range(BLOCK_P):
        outer_id = pid.to(tl.int32) * BLOCK_P + row_offset
        row_mask = n_mask & (outer_id < outer_size_i32)
        offsets = outer_id * dim_size_i32 + n_offsets
        original = tl.load(inp + offsets, mask=row_mask, other=0)
        fill_values = tl.full([BLOCK_N], value_scalar, dtype=original.dtype)
        result = tl.where(selected, fill_values, original)
        tl.store(out + offsets, result, mask=row_mask)


def _generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    return code


def _generate_strided_kernel(
    rank: int,
    dim: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("out,")
        code.writeline("index,")
        code.writeline("value,")
        code.writeline("N,")
        code.writeline("index_len,")
        code.writeline("dim_size,")
        code.writeline(", ".join(f"shape_{i}: int" for i in range(rank)) + ",")
        code.writeline(", ".join(f"stride_{i}: int" for i in range(rank)) + ",")
        code.writeline("VALUE_IS_TENSOR: tl.constexpr,")
        code.writeline("BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(axis=0)")
        code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < N")
        code.writeline("linear = offsets.to(tl.int64)")
        code.writeline("out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)")
        code.newline()

        for i in range(rank - 1, -1, -1):
            logical_size = "index_len" if i == dim else f"shape_{i}"
            code.writeline(f"coord_{i} = linear % {logical_size}")
            if i != 0:
                code.writeline(f"linear = linear // {logical_size}")
            if i == dim:
                code.writeline(
                    f"raw_index = tl.load(index + coord_{i}, mask=mask, other=0)"
                    ".to(tl.int64)"
                )
                code.writeline(
                    "valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)"
                )
                code.writeline(
                    f"coord_{i} = tl.where("
                    "raw_index < 0, raw_index + dim_size, raw_index)"
                )
            code.writeline(f"out_offsets += coord_{i} * stride_{i}")

        code.newline()
        code.writeline('tl.device_assert((~mask) | valid_index, "index out of bounds")')
        code.writeline("store_mask = mask & valid_index")
        code.writeline("if VALUE_IS_TENSOR:")
        with code.indent():
            code.writeline("fill_value = tl.load(value)")
        code.writeline("else:")
        with code.indent():
            code.writeline("fill_value = value")
        code.writeline("tl.store(out + out_offsets, fill_value, mask=store_mask)")

    code.newline()
    return code


def _generate_strided_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(
        f"def {wrapper_name}("
        "out, dim, index, value, N, index_len, dim_size, value_is_tensor):"
    )
    with code.indent():
        code.writeline("out_shapes = list(out.shape)")
        code.writeline("out_strides = list(out.stride())")
        code.writeline("BLOCK_SIZE = 512")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("out,")
            code.writeline("index,")
            code.writeline("value,")
            code.writeline("N,")
            code.writeline("index_len,")
            code.writeline("dim_size,")
            code.writeline(", ".join(f"out_shapes[{i}]" for i in range(rank)) + ",")
            code.writeline(", ".join(f"out_strides[{i}]" for i in range(rank)) + ",")
            code.writeline("VALUE_IS_TENSOR=value_is_tensor,")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE,")
        code.writeline(")")
        code.writeline("return out")

    return code


def _generate_strided_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    out = inputs[0]
    dim = inputs[1]
    rank = out.ndim
    code = _generate_imports(code)
    code = _generate_strided_kernel(rank, dim, kernel_name, code)
    return _generate_strided_wrapper(rank, wrapper_name, kernel_name, code)


class _AscendStridedIndexFillFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self._arg_key(*args)
        if key in self.overloads:
            return self.overloads[key](*args, **kwargs)

        code = IndentedBuffer()
        code = _generate_strided_code(
            args,
            "_ascend_index_fill_wrapper",
            "_ascend_index_fill_kernel",
            code,
        )
        file_name = f"ascend_index_fill_{key}_pid_{self.pid}.py"
        file_path = code_cache_dir() / file_name
        write_atomic(file_path, code.getvalue())

        spec = importlib.util.spec_from_file_location(
            f"_ascend_index_fill_{key}_pid_{self.pid}",
            file_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load generated Ascend index_fill kernel")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        overload = getattr(module, "_ascend_index_fill_wrapper")
        self.overloads[key] = overload
        return overload(*args, **kwargs)

    @staticmethod
    def _arg_key(*args):
        out = args[0]
        dim = args[1]
        return f"rank_{out.ndim}_dim_{dim}"


_strided_index_fill = _AscendStridedIndexFillFunction()


_ASCEND_INDEX_HOST_CHECK_MAX_BYTES = 32 * 1024
_ASCEND_INDEX_VALIDATION_CACHE_MAX_ENTRIES = 256
_ASCEND_MEMBERSHIP_CACHE_MAX_ENTRIES = 64
_ASCEND_MEMBERSHIP_CACHE_MAX_BYTES = 16 * 1024 * 1024
_ascend_membership_cache = {}
_ascend_membership_cache_bytes = 0
_ascend_index_validation_cache = {}


def _should_use_ascend_host_index_check(index):
    return index.numel() * index.element_size() <= _ASCEND_INDEX_HOST_CHECK_MAX_BYTES


def _get_ascend_index_version(index):
    try:
        return index._version
    except RuntimeError:
        # Inference tensors do not expose a version counter, so they cannot
        # safely participate in a mutation-sensitive validation cache.
        return None


def _get_cached_ascend_index_bounds(index, dim_size):
    cache_key = id(index)
    entry = _ascend_index_validation_cache.get(cache_key)
    if entry is None:
        return None

    index_ref, version, cached_dim_size, has_negative, host_index = entry
    if (
        index_ref() is index
        and version == _get_ascend_index_version(index)
        and cached_dim_size == dim_size
    ):
        return has_negative, host_index

    _ascend_index_validation_cache.pop(cache_key, None)
    return None


def _cache_ascend_index_bounds(index, dim_size, has_negative, host_index):
    version = _get_ascend_index_version(index)
    if version is None:
        return

    cache_key = id(index)

    def _remove_entry(index_ref):
        entry = _ascend_index_validation_cache.get(cache_key)
        if entry is not None and entry[0] is index_ref:
            _ascend_index_validation_cache.pop(cache_key, None)

    try:
        index_ref = weakref.ref(index, _remove_entry)
    except TypeError:
        return

    _ascend_index_validation_cache[cache_key] = (
        index_ref,
        version,
        dim_size,
        has_negative,
        host_index,
    )
    while (
        len(_ascend_index_validation_cache) > _ASCEND_INDEX_VALIDATION_CACHE_MAX_ENTRIES
    ):
        _ascend_index_validation_cache.pop(next(iter(_ascend_index_validation_cache)))


def _pop_cached_ascend_membership_mask(cache_key):
    global _ascend_membership_cache_bytes

    entry = _ascend_membership_cache.pop(cache_key, None)
    if entry is not None:
        _ascend_membership_cache_bytes -= entry[-1]
    return entry


def _get_cached_ascend_membership_mask(index, dim_size, device):
    cache_key = (id(index), device)
    entry = _ascend_membership_cache.get(cache_key)
    if entry is None:
        return None

    index_ref, version, cached_dim_size, membership, _ = entry
    if (
        index_ref() is index
        and version == _get_ascend_index_version(index)
        and cached_dim_size == dim_size
    ):
        _ascend_membership_cache.pop(cache_key)
        _ascend_membership_cache[cache_key] = entry
        return membership

    _pop_cached_ascend_membership_mask(cache_key)
    return None


def _cache_ascend_membership_mask(index, dim_size, membership):
    global _ascend_membership_cache_bytes

    version = _get_ascend_index_version(index)
    membership_bytes = membership.numel() * membership.element_size()
    if version is None or membership_bytes > _ASCEND_MEMBERSHIP_CACHE_MAX_BYTES:
        return

    cache_key = (id(index), str(membership.device))

    def _remove_entry(index_ref):
        entry = _ascend_membership_cache.get(cache_key)
        if entry is not None and entry[0] is index_ref:
            _pop_cached_ascend_membership_mask(cache_key)

    try:
        index_ref = weakref.ref(index, _remove_entry)
    except TypeError:
        return

    _pop_cached_ascend_membership_mask(cache_key)
    while _ascend_membership_cache and (
        len(_ascend_membership_cache) >= _ASCEND_MEMBERSHIP_CACHE_MAX_ENTRIES
        or _ascend_membership_cache_bytes + membership_bytes
        > _ASCEND_MEMBERSHIP_CACHE_MAX_BYTES
    ):
        _pop_cached_ascend_membership_mask(next(iter(_ascend_membership_cache)))

    _ascend_membership_cache[cache_key] = (
        index_ref,
        version,
        dim_size,
        membership,
        membership_bytes,
    )
    _ascend_membership_cache_bytes += membership_bytes


def _check_ascend_index_bounds(index, dim_size):
    cached = _get_cached_ascend_index_bounds(index, dim_size)
    if cached is not None:
        return cached

    host_index = None
    if _should_use_ascend_host_index_check(index):
        # A single small D2H copy avoids a device reduction and two scalar transfers.
        host_index = index.cpu()
        min_index, max_index = torch.aminmax(host_index)
    else:
        min_index, max_index = torch.aminmax(index)
    min_index = int(min_index.item())
    max_index = int(max_index.item())
    if min_index < -dim_size or max_index >= dim_size:
        raise IndexError("index out of range in self")
    result = min_index < 0, host_index
    _cache_ascend_index_bounds(index, dim_size, *result)
    return result


def _prepare_ascend_index(inp, dim, index):
    dim, index = _prepare_index(inp, dim, index)
    bounds_checked = False
    has_negative = False
    host_index = None
    if index.numel() > 0:
        has_negative, host_index = _check_ascend_index_bounds(index, inp.size(dim))
        bounds_checked = True
    return dim, index.contiguous(), bounds_checked, has_negative, host_index


def _get_contiguous_config(inner_size):
    block_n = min(64, triton.next_power_of_2(inner_size))
    block_m = max(1, 256 // block_n)
    return block_m, block_n


def _get_inner1_config(index_len):
    return min(512, max(128, triton.next_power_of_2(index_len)))


def _get_inner1_membership_mask_config(outer_size, dim_size):
    block_n = min(4096, triton.next_power_of_2(dim_size))
    n_tiles = triton.cdiv(dim_size, block_n)
    outer_blocks = max(1, triton.cdiv(32, n_tiles))
    block_p = min(512, triton.next_power_of_2(triton.cdiv(outer_size, outer_blocks)))
    block_p = max(1, block_p)
    return block_n, block_p


def _use_int32_indexing(out, dim_size, index_len):
    int32_max = 2**31 - 1
    return out.numel() <= int32_max and dim_size <= int32_max and index_len <= int32_max


def _should_use_contiguous_membership_mask(
    out, index, bounds_checked, outer_size, inner_size
):
    if not bounds_checked or index.numel() <= 8:
        return False
    if out.numel() < 1 << 20:
        return False
    if inner_size != 1 or out.numel() > 2**31 - 1:
        return False
    return outer_size > 1


def _should_use_fused_membership_build(dim_size, index_len):
    return dim_size <= 4096 and index_len <= 256


def _can_use_contiguous_dim0_rows(out, dim, index, bounds_checked):
    if (
        not bounds_checked
        or not out.is_contiguous()
        or index.numel() == 0
        or out.numel() > 2**31 - 1
    ):
        return False

    dim_size = out.size(dim)
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = out.numel() // (dim_size * inner_size)
    return outer_size == 1


def _can_use_contiguous_small_inner_updates(
    out, dim, index, value_is_tensor, bounds_checked
):
    index_len = index.numel()
    if (
        not bounds_checked
        or value_is_tensor
        or not out.is_contiguous()
        or (index_len != 1 and index_len < 32)
        or out.numel() > 2**31 - 1
    ):
        return False

    dim_size = out.size(dim)
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = out.numel() // (dim_size * inner_size)
    return (
        outer_size > 1
        and inner_size <= 4
        and (index_len == 1 or (inner_size > 1 and index_len >= 32))
    )


def _can_use_contiguous_high_density_transpose_fill(
    out, dim, index, value_is_tensor, bounds_checked
):
    if (
        not bounds_checked
        or value_is_tensor
        or not out.is_contiguous()
        or out.numel() > 2**31 - 1
    ):
        return False

    dim_size = out.size(dim)
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = out.numel() // (dim_size * inner_size)
    estimated_sparse_programs = math.ceil(
        index.numel() / _SMALL_INNER_BLOCK_I
    ) * math.ceil(outer_size / _SMALL_INNER_BLOCK_OUTER)
    is_wide_dense = dim_size >= 4096 and index.numel() * 2 >= dim_size - 1
    is_small_full_dim = (
        dim_size <= _TRANSPOSE_FILL_SMALL_FULL_DIM_MAX_SIZE
        and out.numel() >= _TRANSPOSE_FILL_SMALL_FULL_DIM_MIN_NUMEL
        and index.numel() >= dim_size
    )
    return (
        outer_size > 1
        and 1 <= inner_size <= 4
        and outer_size * inner_size >= 256
        and (
            is_wide_dense
            or is_small_full_dim
            or (
                dim_size >= 4096
                and inner_size > 1
                and estimated_sparse_programs >= _TRANSPOSE_FILL_MIN_SPARSE_PROGRAMS
            )
        )
    )


def _index_fill_contiguous_small_inner_updates(out, dim, index, value, has_negative):
    dim_size = out.size(dim)
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = out.numel() // (dim_size * inner_size)
    block_outer = _SMALL_INNER_BLOCK_OUTER

    with torch_device_fn.device(out.device):
        if 2 <= inner_size <= 4:
            grid = (index.numel(), triton.cdiv(outer_size, block_outer))
            index_fill_contiguous_scalar_small_inner_blockptr_kernel[grid](
                out,
                index,
                value,
                outer_size,
                dim_size,
                inner_size,
                HAS_NEGATIVE=has_negative,
                BLOCK_OUTER=block_outer,
                SPAN=inner_size,
            )
        else:
            block_i = _SMALL_INNER_BLOCK_I
            block_n = _SMALL_INNER_BLOCK_N
            grid = (
                triton.cdiv(index.numel(), block_i),
                triton.cdiv(outer_size, block_outer),
            )
            index_fill_contiguous_scalar_small_inner_kernel[grid](
                out,
                index,
                value,
                outer_size,
                index.numel(),
                dim_size,
                inner_size,
                HAS_NEGATIVE=has_negative,
                BLOCK_I=block_i,
                BLOCK_OUTER=block_outer,
                BLOCK_N=block_n,
            )
    return out


def _can_use_contiguous_membership_mask(out, dim, index, bounds_checked):
    if not out.is_contiguous():
        return False
    dim_size = out.size(dim)
    inner_size = 1
    for size in out.shape[dim + 1 :]:
        inner_size *= size
    outer_size = out.numel() // (dim_size * inner_size)
    return _should_use_contiguous_membership_mask(
        out, index, bounds_checked, outer_size, inner_size
    )


def _build_contiguous_membership_mask(out, index, has_negative, dim_size):
    cached = _get_cached_ascend_membership_mask(index, dim_size, str(out.device))
    if cached is not None:
        return cached

    index_len = index.numel()
    block_i = 256
    marker_grid = (min(triton.cdiv(index_len, block_i), 128),)
    use_int32 = _use_int32_indexing(out, dim_size, index_len)

    with torch_device_fn.device(out.device):
        if _should_use_fused_membership_build(dim_size, index_len):
            membership = torch.empty((dim_size,), dtype=torch.int32, device=out.device)
            index_fill_membership_mask_build_kernel[(1,)](
                membership,
                index,
                index_len,
                dim_size,
                HAS_NEGATIVE=has_negative,
                USE_INT32=use_int32,
                BLOCK_N=1024,
                BLOCK_I=block_i,
            )
        else:
            membership = torch.zeros((dim_size,), dtype=torch.int32, device=out.device)
            index_fill_membership_mask_kernel[marker_grid](
                membership,
                index,
                index_len,
                dim_size,
                HAS_NEGATIVE=has_negative,
                USE_INT32=use_int32,
                BLOCK_I=block_i,
            )
    _cache_ascend_membership_mask(index, dim_size, membership)
    return membership


def _has_full_contiguous_index_coverage(index, dim_size, host_index=None):
    if host_index is None:
        host_index = index.cpu()
    selected = np.zeros(dim_size, dtype=np.bool_)
    # Bounds validation makes NumPy's negative indexing exactly match normalization.
    selected[host_index.numpy()] = True
    return bool(selected.all())


def _can_try_contiguous_full_coverage_fill(
    out, dim, index, value_is_tensor, bounds_checked
):
    if (
        not bounds_checked
        or value_is_tensor
        or not out.is_contiguous()
        or index.numel() != out.size(dim)
        or (index.numel() * index.element_size() > _FULL_COVERAGE_HOST_CHECK_MAX_BYTES)
        or out.numel() > 2**31 - 1
    ):
        return False
    inner_size = math.prod(out.shape[dim + 1 :])
    outer_size = out.numel() // (out.size(dim) * inner_size)
    return outer_size > 1 and 1 <= inner_size <= 4


def _try_index_fill_contiguous_full_coverage_fill(
    inp, dim, index, value, value_is_tensor, inplace, host_index=None
):
    dim_size = inp.size(dim)
    if not _has_full_contiguous_index_coverage(index, dim_size, host_index):
        return None

    out = inp if inplace else torch.empty_like(inp)
    with torch_device_fn.device(inp.device):
        index_fill_contiguous_full_kernel[(triton.cdiv(inp.numel(), 4096),)](
            out,
            value,
            inp.numel(),
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK_SIZE=4096,
        )
    return out


def _index_fill_contiguous_dim0_rows(
    out, dim, index, value, value_is_tensor, has_negative
):
    dim_size = out.size(dim)
    inner_size = math.prod(out.shape[dim + 1 :])
    grid = (index.numel(),)
    with torch_device_fn.device(out.device):
        index_fill_contiguous_dim0_rows_kernel[grid](
            out,
            index,
            value,
            index.numel(),
            dim_size,
            inner_size,
            HAS_NEGATIVE=has_negative,
            USE_INT32=_use_int32_indexing(out, dim_size, index.numel()),
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK_N=4096,
        )
    return out


def _index_fill_contiguous_dim0_rows_functional(
    inp, dim, index, value, value_is_tensor, has_negative
):
    out = _native_clone(inp)
    return _index_fill_contiguous_dim0_rows(
        out, dim, index, value, value_is_tensor, has_negative
    )


def _index_fill_contiguous_high_density_transpose_fill(
    inp, dim, index, value, value_is_tensor, has_negative, inplace
):
    dim_size = inp.size(dim)
    inner_size = math.prod(inp.shape[dim + 1 :])
    outer_size = inp.numel() // (dim_size * inner_size)

    # Materialize [dim, outer, inner] so each selected dim value is one
    # contiguous row rather than outer_size disjoint small-inner writes.
    transposed = (
        inp.reshape(outer_size, dim_size, inner_size).permute(1, 0, 2).contiguous()
    )
    _index_fill_contiguous_dim0_rows(
        transposed, 0, index, value, value_is_tensor, has_negative
    )
    restored = transposed.permute(1, 0, 2).reshape_as(inp)
    if inplace:
        inp.copy_(restored)
        return inp
    return restored.contiguous()


def _index_fill_contiguous_membership_mask(
    out,
    index,
    value,
    value_is_tensor,
    has_negative,
    outer_size,
    dim_size,
    inner_size,
    source=None,
):
    block_n, block_p = _get_inner1_membership_mask_config(outer_size, dim_size)
    select_grid = (triton.cdiv(dim_size, block_n) * triton.cdiv(outer_size, block_p),)
    membership = _build_contiguous_membership_mask(out, index, has_negative, dim_size)

    with torch_device_fn.device(out.device):
        if source is None:
            index_fill_contiguous_mask_inner1_reuse_kernel[select_grid](
                out,
                membership,
                value,
                outer_size,
                dim_size,
                VALUE_IS_TENSOR=value_is_tensor,
                BLOCK_N=block_n,
                BLOCK_P=block_p,
            )
        else:
            index_fill_contiguous_mask_inner1_copy_reuse_kernel[select_grid](
                source,
                out,
                membership,
                value,
                outer_size,
                dim_size,
                VALUE_IS_TENSOR=value_is_tensor,
                BLOCK_N=block_n,
                BLOCK_P=block_p,
            )
    return out


def _should_use_local_membership_mask(dim_size):
    return dim_size <= 256


def _index_fill_contiguous_local_membership_mask(
    source,
    out,
    index,
    value,
    value_is_tensor,
    has_negative,
    outer_size,
    dim_size,
):
    index_len = index.numel()
    block_i = 32
    block_n = min(256, triton.next_power_of_2(dim_size))
    _, block_p = _get_inner1_membership_mask_config(outer_size, dim_size)
    grid = (triton.cdiv(outer_size, block_p),)
    use_int32 = _use_int32_indexing(out, dim_size, index_len)

    with torch_device_fn.device(out.device):
        index_fill_contiguous_local_membership_inner1_kernel[grid](
            source,
            out,
            index,
            value,
            index_len,
            outer_size,
            dim_size,
            HAS_NEGATIVE=has_negative,
            USE_INT32=use_int32,
            VALUE_IS_TENSOR=value_is_tensor,
            BLOCK_I=block_i,
            BLOCK_N=block_n,
            BLOCK_P=block_p,
        )
    return out


def _index_fill_contiguous_inner1(
    out,
    dim,
    index,
    value,
    value_is_tensor,
    has_negative,
):
    dim_size = out.size(dim)
    outer_size = out.numel() // dim_size
    index_len = index.numel()
    block_i = _get_inner1_config(index_len)
    grid = (outer_size, triton.cdiv(index_len, block_i))
    use_int32 = _use_int32_indexing(out, dim_size, index_len)

    with torch_device_fn.device(out.device):
        if value_is_tensor:
            index_fill_contiguous_tensor_inner1_kernel[grid](
                out,
                index,
                value,
                index_len,
                dim_size,
                HAS_NEGATIVE=has_negative,
                USE_INT32=use_int32,
                BLOCK_I=block_i,
            )
        else:
            index_fill_contiguous_scalar_inner1_kernel[grid](
                out,
                index,
                value,
                index_len,
                dim_size,
                HAS_NEGATIVE=has_negative,
                USE_INT32=use_int32,
                BLOCK_I=block_i,
            )
    return out


def _index_fill_contiguous(
    out,
    dim,
    index,
    value,
    value_is_tensor,
    bounds_checked,
    has_negative,
    host_index=None,
):
    dim_size = out.size(dim)
    inner_size = 1
    for size in out.shape[dim + 1 :]:
        inner_size *= size
    outer_size = out.numel() // (dim_size * inner_size)
    if _can_use_contiguous_dim0_rows(out, dim, index, bounds_checked):
        return _index_fill_contiguous_dim0_rows(
            out, dim, index, value, value_is_tensor, has_negative
        )
    if _can_try_contiguous_full_coverage_fill(
        out, dim, index, value_is_tensor, bounds_checked
    ):
        full_fill = _try_index_fill_contiguous_full_coverage_fill(
            out, dim, index, value, value_is_tensor, inplace=True, host_index=host_index
        )
        if full_fill is not None:
            return full_fill
    if _can_use_contiguous_high_density_transpose_fill(
        out, dim, index, value_is_tensor, bounds_checked
    ):
        return _index_fill_contiguous_high_density_transpose_fill(
            out, dim, index, value, value_is_tensor, has_negative, inplace=True
        )
    if _can_use_contiguous_small_inner_updates(
        out, dim, index, value_is_tensor, bounds_checked
    ):
        return _index_fill_contiguous_small_inner_updates(
            out, dim, index, value, has_negative
        )
    if _should_use_contiguous_membership_mask(
        out, index, bounds_checked, outer_size, inner_size
    ):
        if _should_use_local_membership_mask(dim_size):
            return _index_fill_contiguous_local_membership_mask(
                out,
                out,
                index,
                value,
                value_is_tensor,
                has_negative,
                outer_size,
                dim_size,
            )
        return _index_fill_contiguous_membership_mask(
            out,
            index,
            value,
            value_is_tensor,
            has_negative,
            outer_size,
            dim_size,
            inner_size,
        )
    if inner_size == 1 and bounds_checked:
        return _index_fill_contiguous_inner1(
            out,
            dim,
            index,
            value,
            value_is_tensor,
            has_negative,
        )
    outer_index_len = outer_size * index.numel()
    block_m, block_n = _get_contiguous_config(inner_size)
    grid = (
        triton.cdiv(outer_index_len, block_m),
        triton.cdiv(inner_size, block_n),
    )

    with torch_device_fn.device(out.device):
        if value_is_tensor:
            index_fill_contiguous_tensor_kernel[grid](
                out,
                index,
                value,
                outer_index_len,
                index.numel(),
                dim_size,
                inner_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        else:
            index_fill_contiguous_scalar_kernel[grid](
                out,
                index,
                value,
                outer_index_len,
                index.numel(),
                dim_size,
                inner_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
    return out


def _index_fill_strided(out, dim, index, value, value_is_tensor):
    dim_size = out.size(dim)
    fill_numel = out.numel() // dim_size * index.numel()
    with torch_device_fn.device(out.device):
        _strided_index_fill(
            out,
            dim,
            index,
            value,
            fill_numel,
            index.numel(),
            dim_size,
            value_is_tensor,
        )
    return out


def _index_fill_impl(
    out,
    dim,
    index,
    value,
    value_is_tensor,
    bounds_checked,
    has_negative,
    host_index=None,
):
    if out.numel() == 0 or index.numel() == 0:
        return out
    if out.is_contiguous():
        return _index_fill_contiguous(
            out,
            dim,
            index,
            value,
            value_is_tensor,
            bounds_checked,
            has_negative,
            host_index,
        )
    return _index_fill_strided(out, dim, index, value, value_is_tensor)


def _index_fill_functional(
    inp,
    dim,
    index,
    value,
    value_is_tensor,
    bounds_checked,
    has_negative,
    host_index=None,
):
    if _can_use_contiguous_dim0_rows(inp, dim, index, bounds_checked):
        return _index_fill_contiguous_dim0_rows_functional(
            inp, dim, index, value, value_is_tensor, has_negative
        )
    if _can_try_contiguous_full_coverage_fill(
        inp, dim, index, value_is_tensor, bounds_checked
    ):
        full_fill = _try_index_fill_contiguous_full_coverage_fill(
            inp,
            dim,
            index,
            value,
            value_is_tensor,
            inplace=False,
            host_index=host_index,
        )
        if full_fill is not None:
            return full_fill
    if _can_use_contiguous_high_density_transpose_fill(
        inp, dim, index, value_is_tensor, bounds_checked
    ):
        return _index_fill_contiguous_high_density_transpose_fill(
            inp, dim, index, value, value_is_tensor, has_negative, inplace=False
        )
    if _can_use_contiguous_membership_mask(inp, dim, index, bounds_checked):
        out = torch.empty_like(inp)
        outer_size = out.numel() // out.size(dim)
        if _should_use_local_membership_mask(out.size(dim)):
            return _index_fill_contiguous_local_membership_mask(
                inp,
                out,
                index,
                value,
                value_is_tensor,
                has_negative,
                outer_size,
                out.size(dim),
            )
        return _index_fill_contiguous_membership_mask(
            out,
            index,
            value,
            value_is_tensor,
            has_negative,
            outer_size,
            out.size(dim),
            1,
            source=inp,
        )

    out = _native_clone(inp)
    return _index_fill_impl(
        out,
        dim,
        index,
        value,
        value_is_tensor,
        bounds_checked,
        has_negative,
        host_index,
    )


def index_fill(inp, dim, index, value):
    # Entry for both `index_fill.int_Scalar` and `index_fill.int_Tensor`: the
    # dispatcher routes by value type, so a 0-dimensional tensor value arrives
    # as a Tensor and a plain number as a Python scalar.
    logger.debug("GEMS_ASCEND INDEX_FILL")
    dim, index, bounds_checked, has_negative, host_index = _prepare_ascend_index(
        inp, dim, index
    )
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    return _index_fill_functional(
        inp,
        dim,
        index,
        value,
        value_is_tensor,
        bounds_checked,
        has_negative,
        host_index,
    )


def index_fill_(inp, dim, index, value):
    # Entry for both `index_fill_.int_Scalar` and `index_fill_.int_Tensor`.
    logger.debug("GEMS_ASCEND INDEX_FILL_")
    dim, index, bounds_checked, has_negative, host_index = _prepare_ascend_index(
        inp, dim, index
    )
    if isinstance(value, torch.Tensor):
        value_is_tensor, value = _prepare_tensor_value(inp, value)
    else:
        value_is_tensor = False
    return _index_fill_impl(
        inp,
        dim,
        index,
        value,
        value_is_tensor,
        bounds_checked,
        has_negative,
        host_index,
    )
