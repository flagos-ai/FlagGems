import importlib.util
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_fill import (
    _index_fill_uses_device_bounds_check,
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
    normalized_index = tl.where(
        raw_index < 0, raw_index + dim_size, raw_index
    ).to(tl.int64)

    out_offsets = outer_coord[:, None].to(tl.int64) * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]

    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask &= valid_index[:, None]
    tl.store(out + out_offsets, value, mask=store_mask)


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
    normalized_index = tl.where(
        raw_index < 0, raw_index + dim_size, raw_index
    ).to(tl.int64)

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
@triton.jit(
    do_not_specialize=["outer_size", "dim_size"]
)
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
        code.writeline(
            'tl.device_assert((~mask) | valid_index, "index out of bounds")'
        )
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
            code.writeline(
                ", ".join(f"out_strides[{i}]" for i in range(rank)) + ","
            )
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


def _check_ascend_index_bounds(index, dim_size):
    min_index, max_index = torch.aminmax(index)
    min_index = int(min_index.item())
    max_index = int(max_index.item())
    if min_index < -dim_size or max_index >= dim_size:
        raise IndexError("index out of range in self")
    return min_index < 0


def _prepare_ascend_index(inp, dim, index):
    dim, index = _prepare_index(inp, dim, index)
    bounds_checked = False
    has_negative = False
    if index.numel() > 0 and _index_fill_uses_device_bounds_check():
        has_negative = _check_ascend_index_bounds(index, inp.size(dim))
        bounds_checked = True
    return dim, index.contiguous(), bounds_checked, has_negative


def _get_contiguous_config(inner_size):
    block_n = min(64, triton.next_power_of_2(inner_size))
    block_m = max(1, 256 // block_n)
    return block_m, block_n


def _get_inner1_config(index_len):
    return min(512, max(128, triton.next_power_of_2(index_len)))


def _get_inner1_membership_mask_config(outer_size, dim_size):
    block_n = min(1024, triton.next_power_of_2(dim_size))
    n_tiles = triton.cdiv(dim_size, block_n)
    outer_blocks = max(1, triton.cdiv(32, n_tiles))
    block_p = min(512, triton.next_power_of_2(triton.cdiv(outer_size, outer_blocks)))
    block_p = max(1, block_p)
    return block_n, block_p


def _use_int32_indexing(out, dim_size, index_len):
    int32_max = 2**31 - 1
    return (
        out.numel() <= int32_max
        and dim_size <= int32_max
        and index_len <= int32_max
    )


def _should_use_contiguous_membership_mask(
    out, index, bounds_checked, outer_size, inner_size
):
    if not bounds_checked or index.numel() <= 16:
        return False
    if out.numel() < 1 << 20:
        return False
    if inner_size != 1 or out.numel() > 2**31 - 1:
        return False
    return outer_size > 1


def _index_fill_contiguous_membership_mask(
    out,
    index,
    value,
    value_is_tensor,
    has_negative,
    outer_size,
    dim_size,
    inner_size,
):
    index_len = index.numel()
    block_i = 256
    block_n, block_p = _get_inner1_membership_mask_config(outer_size, dim_size)
    marker_grid = (min(triton.cdiv(index_len, block_i), 128),)
    select_grid = (
        triton.cdiv(dim_size, block_n) * triton.cdiv(outer_size, block_p),
    )
    use_int32 = _use_int32_indexing(out, dim_size, index_len)

    with torch_device_fn.device(out.device):
        membership = torch.zeros(
            (dim_size,), dtype=torch.int32, device=out.device
        )
        index_fill_membership_mask_kernel[marker_grid](
            membership,
            index,
            index_len,
            dim_size,
            HAS_NEGATIVE=has_negative,
            USE_INT32=use_int32,
            BLOCK_I=block_i,
        )
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
):
    dim_size = out.size(dim)
    inner_size = 1
    for size in out.shape[dim + 1 :]:
        inner_size *= size
    outer_size = out.numel() // (dim_size * inner_size)
    if _should_use_contiguous_membership_mask(
        out, index, bounds_checked, outer_size, inner_size
    ):
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
):
    if out.numel() == 0 or index.numel() == 0:
        return out
    if out.is_contiguous():
        return _index_fill_contiguous(
            out, dim, index, value, value_is_tensor, bounds_checked, has_negative
        )
    return _index_fill_strided(out, dim, index, value, value_is_tensor)


def index_fill_scalar(inp, dim, index, value):
    logger.debug("GEMS_ASCEND INDEX_FILL SCALAR")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    out = _native_clone(inp)
    return _index_fill_impl(
        out, dim, index, value, False, bounds_checked, has_negative
    )


def index_fill_tensor(inp, dim, index, value):
    logger.debug("GEMS_ASCEND INDEX_FILL TENSOR")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    out = _native_clone(inp)
    return _index_fill_impl(
        out, dim, index, value, value_is_tensor, bounds_checked, has_negative
    )


def index_fill_scalar_out(inp, dim, index, value, *, out):
    logger.debug("GEMS_ASCEND INDEX_FILL SCALAR_OUT")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    out.copy_(inp)
    return _index_fill_impl(
        out, dim, index, value, False, bounds_checked, has_negative
    )


def index_fill_tensor_out(inp, dim, index, value, *, out):
    logger.debug("GEMS_ASCEND INDEX_FILL TENSOR_OUT")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    out.copy_(inp)
    return _index_fill_impl(
        out, dim, index, value, value_is_tensor, bounds_checked, has_negative
    )


def index_fill_scalar_(inp, dim, index, value):
    logger.debug("GEMS_ASCEND INDEX_FILL_ SCALAR")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    return _index_fill_impl(
        inp, dim, index, value, False, bounds_checked, has_negative
    )


def index_fill_tensor_(inp, dim, index, value):
    logger.debug("GEMS_ASCEND INDEX_FILL_ TENSOR")
    dim, index, bounds_checked, has_negative = _prepare_ascend_index(
        inp, dim, index
    )
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    return _index_fill_impl(
        inp, dim, index, value, value_is_tensor, bounds_checked, has_negative
    )
