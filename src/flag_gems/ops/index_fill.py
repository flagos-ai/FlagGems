import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def _generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    code.newline()
    return code


def _generate_index_fill_kernel(
    rank: int,
    dim: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("index,")
        code.writeline("value_tensor,")
        code.writeline("out,")
        code.writeline("index_numel,")
        code.writeline("slice_numel,")
        code.writeline("dim_size,")
        code.writeline("dim_stride,")
        code.writeline("value,")
        code.writeline("HAS_VALUE_TENSOR: tl.constexpr,")
        stride_args = ", ".join(f"out_stride_{i}: int" for i in range(rank))
        code.writeline(f"{stride_args},")
        shape_args = ", ".join(f"out_shape_{i}: int" for i in range(rank))
        code.writeline(f"{shape_args},")
        code.writeline("BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline(
            "offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"
        )
        code.writeline("total = index_numel * slice_numel")
        code.writeline("mask = offsets < total")
        code.writeline("index_pos = offsets // slice_numel")
        code.writeline("remaining = offsets - index_pos * slice_numel")
        code.writeline("out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)")
        for axis in range(rank - 1, -1, -1):
            if axis == dim:
                continue
            code.writeline(f"coord_{axis} = remaining % out_shape_{axis}")
            code.writeline(f"remaining = remaining // out_shape_{axis}")
            code.writeline(f"out_offsets += coord_{axis} * out_stride_{axis}")
        code.writeline(
            "index_values = tl.load(index + index_pos, mask=mask, other=0).to(tl.int64)"
        )
        code.writeline(
            "index_values = tl.where(index_values < 0, index_values + dim_size, index_values)"
        )
        code.writeline("out_offsets += index_values * dim_stride")
        code.writeline("fill_value = value")
        code.writeline("if HAS_VALUE_TENSOR:")
        with code.indent():
            code.writeline("fill_value = tl.load(value_tensor)")
        code.writeline("tl.store(out + out_offsets, fill_value, mask=mask)")
    code.newline()
    code.newline()
    return code


def _generate_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(
        f"def {wrapper_name}(index, value_tensor, out, dim, index_numel, slice_numel, "
        "dim_size, dim_stride, value, has_value_tensor):"
    )
    with code.indent():
        code.writeline("out_strides = list(out.stride())")
        code.writeline("out_shapes = list(out.shape)")
        code.writeline("BLOCK_SIZE = 256")
        code.writeline("grid = (triton.cdiv(index_numel * slice_numel, BLOCK_SIZE),)")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("index,")
            code.writeline("value_tensor,")
            code.writeline("out,")
            code.writeline("index_numel,")
            code.writeline("slice_numel,")
            code.writeline("dim_size,")
            code.writeline("dim_stride,")
            code.writeline("value,")
            for axis in range(rank):
                code.writeline(f"out_strides[{axis}],")
            for axis in range(rank):
                code.writeline(f"out_shapes[{axis}],")
            code.writeline("HAS_VALUE_TENSOR=has_value_tensor,")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE,")
        code.writeline(")")
        code.writeline("return out")
    return code


def _generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    out = inputs[2]
    dim = inputs[3]
    rank = out.ndim

    code = _generate_imports(code)
    code = _generate_index_fill_kernel(rank, dim, kernel_name, code)
    code = _generate_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexFillFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        overload = self.overloads.get(key)
        if overload is None:
            code = IndentedBuffer()
            code = _generate_code(
                args,
                "_index_fill_wrapper",
                "_index_fill_jit_function",
                code,
            )
            file_name = f"index_fill_rank_dim_{key}_pid_{self.pid}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_index_fill_rank_dim_{key}_pid_{self.pid}",
                file_path,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            overload = getattr(module, "_index_fill_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        out = args[2]
        dim = args[3]
        return f"{out.ndim}_{dim}"


_index_fill_func = IndexFillFunction()


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return True


def _normalize_dim(dim: int, ndim: int):
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )
    return dim % ndim


def _check_index_fill_input(inp, dim, index, value):
    if not isinstance(index, torch.Tensor):
        raise TypeError("index_fill_(): argument 'index' must be Tensor")
    if index.dtype != torch.int64:
        raise IndexError("index_fill_(): Expected dtype int64 for index.")
    if index.dim() > 1:
        raise RuntimeError("Index has to be a vector/scalar")
    if index.device != inp.device:
        raise RuntimeError("index_fill_(): index must be on the same device as self")
    if torch.is_tensor(value) and value.dim() != 0:
        raise RuntimeError(
            "index_fill_ only supports a 0-dimensional value tensor, "
            f"but got tensor with {value.dim()} dimension(s)."
        )
    if torch.is_tensor(value) and value.device != inp.device:
        raise RuntimeError("index_fill_(): value must be on the same device as self")

    dim = _normalize_dim(dim, inp.ndim)
    if index.numel() > 0:
        dim_size = inp.size(dim)
        index_for_bounds = index.reshape(-1)
        min_index = index_for_bounds.min().item()
        max_index = index_for_bounds.max().item()
        if min_index < -dim_size or max_index >= dim_size:
            bad_index = min_index if min_index < -dim_size else max_index
            raise IndexError(
                f"index {bad_index} is out of bounds for dimension {dim} with size {dim_size}"
            )
    return dim


def _slice_numel(inp, dim):
    numel = inp.numel()
    dim_size = inp.size(dim)
    if dim_size == 0:
        return 0
    return numel // dim_size


def _launch_index_fill(out, dim, index, value):
    index = index.reshape(-1).contiguous()
    index_numel = index.numel()
    slice_numel = _slice_numel(out, dim)
    if index_numel == 0 or slice_numel == 0:
        return out

    has_value_tensor = torch.is_tensor(value)
    value_tensor = value.contiguous() if has_value_tensor else index
    scalar_value = 0 if has_value_tensor else value

    _index_fill_func(
        index,
        value_tensor,
        out,
        dim,
        index_numel,
        slice_numel,
        out.size(dim),
        out.stride(dim),
        scalar_value,
        has_value_tensor,
    )
    return out


def _index_fill_impl(inp, dim, index, value, out=None, inplace=False):
    logger.debug("GEMS INDEX_FILL")
    dim = _check_index_fill_input(inp, dim, index, value)

    if inplace:
        return _launch_index_fill(inp, dim, index, value)

    if out is None:
        out = inp.clone()
    else:
        if out.dtype != inp.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
            )
        if out.device != inp.device:
            raise RuntimeError("Expected out tensor to be on the same device as self")
        src = inp.clone() if _tensors_overlap(inp, out) else inp
        if tuple(out.shape) != tuple(inp.shape):
            out.resize_(inp.shape)
        out.copy_(src)

    return _launch_index_fill(out, dim, index, value)


def index_fill_scalar(inp, dim, index, value):
    return _index_fill_impl(inp, dim, index, value)


def index_fill_tensor(inp, dim, index, value):
    return _index_fill_impl(inp, dim, index, value)


def index_fill_scalar_(inp, dim, index, value):
    return _index_fill_impl(inp, dim, index, value, inplace=True)


def index_fill_tensor_(inp, dim, index, value):
    return _index_fill_impl(inp, dim, index, value, inplace=True)


def index_fill_scalar_out(inp, dim, index, value, out=None):
    return _index_fill_impl(inp, dim, index, value, out=out)


def index_fill_tensor_out(inp, dim, index, value, out=None):
    return _index_fill_impl(inp, dim, index, value, out=out)
