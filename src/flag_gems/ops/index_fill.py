import importlib
import importlib.abc
import importlib.machinery
import logging
import os
import sys
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

_CPP_INDEX_FILL_SCALAR = None
_CPP_INDEX_FILL_SCALAR_INPLACE = None
_CPP_INDEX_FILL_LOOKED_UP = False


def _get_cpp_index_fill_scalar_inplace():
    global _CPP_INDEX_FILL_SCALAR
    global _CPP_INDEX_FILL_SCALAR_INPLACE, _CPP_INDEX_FILL_LOOKED_UP
    if _CPP_INDEX_FILL_LOOKED_UP:
        return _CPP_INDEX_FILL_SCALAR_INPLACE
    _CPP_INDEX_FILL_LOOKED_UP = True
    try:
        from flag_gems.config import c_operators
    except ImportError:
        c_operators = None
    if c_operators is not None:
        _CPP_INDEX_FILL_SCALAR = getattr(c_operators, "index_fill_scalar", None)
        _CPP_INDEX_FILL_SCALAR_INPLACE = getattr(
            c_operators, "index_fill_scalar_", None
        )
    return _CPP_INDEX_FILL_SCALAR_INPLACE


def _get_cpp_index_fill_scalar():
    _get_cpp_index_fill_scalar_inplace()
    return _CPP_INDEX_FILL_SCALAR


_CPP_INDEX_FILL_ENABLED = os.environ.get(
    "FLAG_GEMS_INDEX_FILL_CPP_LAUNCHER", "1"
).lower() not in ("", "0", "false", "off", "none", "disable", "disabled")


# libtriton_jit builds ASTSource directly and otherwise loses JITFunction.debug.
class _TritonJitCompileProxy:
    def __init__(self, triton_module):
        self._triton_module = triton_module

    def __getattr__(self, name):
        return getattr(self._triton_module, name)

    def compile(self, source, *args, **kwargs):
        if getattr(getattr(source, "fn", None), "debug", False):
            options = dict(kwargs.get("options", {}))
            options["debug"] = True
            kwargs["options"] = options
        return self._triton_module.compile(source, *args, **kwargs)


def _patch_triton_jit_standalone_compile(module):
    if getattr(module, "_flag_gems_debug_proxy", False):
        return
    if not hasattr(module, "compile_a_kernel"):
        return

    triton_module = getattr(module, "triton", None)
    if triton_module is None:
        return

    module.triton = _TritonJitCompileProxy(triton_module)
    module._flag_gems_debug_proxy = True


class _TritonJitStandaloneCompileLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_triton_jit_standalone_compile(module)


class _TritonJitStandaloneCompileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "standalone_compile":
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _TritonJitStandaloneCompileLoader(spec.loader)
        return spec


def _enable_triton_jit_debug_decorator():
    standalone_compile = sys.modules.get("standalone_compile")
    if standalone_compile is not None:
        _patch_triton_jit_standalone_compile(standalone_compile)
        return

    if not any(
        isinstance(finder, _TritonJitStandaloneCompileFinder)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _TritonJitStandaloneCompileFinder())


if _CPP_INDEX_FILL_ENABLED:
    _enable_triton_jit_debug_decorator()


def _cpp_index_fill_enabled():
    return _CPP_INDEX_FILL_ENABLED


def _index_fill_uses_device_bounds_check():
    mode = os.environ.get("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK", "device").lower()
    return mode in (
        "",
        "0",
        "false",
        "off",
        "device",
        "default",
        # Keep former no-host-check spellings as device-check aliases.
        "none",
        "disable",
        "disabled",
    )


def _is_supported_cpp_scalar(value):
    return type(value) in (bool, int, float)


def _index_fill_shape_factors(out, dim):
    dim_size = out.size(dim)
    inner_size = 1
    for size in out.shape[dim + 1 :]:
        inner_size *= size
    outer_size = out.numel() // (dim_size * inner_size)
    return dim_size, inner_size, outer_size


def _should_use_cpp_index_fill(out, dim, index, value):
    if not _is_supported_cpp_scalar(value) or not out.is_contiguous():
        return False

    dim_size, inner_size, outer_size = _index_fill_shape_factors(out, dim)

    # For many tiny rows, the Python path is better once index becomes large.
    if inner_size <= 4 and outer_size > 1 and dim_size > 8192:
        return index.numel() * 16 <= dim_size
    return True


def _should_skip_cpp_index_fill_out(out, dim):
    dim_size, inner_size, outer_size = _index_fill_shape_factors(out, dim)
    return inner_size <= 4 and outer_size > 1 and dim_size > 8192


def _try_cpp_index_fill_scalar_(out, dim, index, value):
    if (
        not _cpp_index_fill_enabled()
        or not _should_use_cpp_index_fill(out, dim, index, value)
    ):
        return None
    cpp_func = _get_cpp_index_fill_scalar_inplace()
    if cpp_func is None:
        return None
    return cpp_func(out, dim, index, value)


def _try_cpp_index_fill_scalar_fast(inp, dim, index, value):
    if (
        not _CPP_INDEX_FILL_ENABLED
        or not _index_fill_uses_device_bounds_check()
        or not _is_supported_cpp_scalar(value)
        or not inp.is_contiguous()
    ):
        return None

    cpp_func = _get_cpp_index_fill_scalar()
    if cpp_func is None:
        return None
    return cpp_func(inp, dim, index, value)


def _try_cpp_index_fill_scalar_fast_(out, dim, index, value):
    if (
        not _CPP_INDEX_FILL_ENABLED
        or not _index_fill_uses_device_bounds_check()
        or not _is_supported_cpp_scalar(value)
        or not out.is_contiguous()
    ):
        return None

    cpp_func = _get_cpp_index_fill_scalar_inplace()
    if cpp_func is None:
        return None
    return cpp_func(out, dim, index, value)


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_clone(inp):
    return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)


def _native_copy_(out, src):
    return torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, src, False)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    code.newline()
    return code


def generate_index_fill_kernel(
    rank: int,
    dim: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit(debug=True)")
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
                raw_index_load = (
                    f"raw_index = tl.load(index + coord_{i}, mask=mask, other=0)"
                    ".to(tl.int64)"
                )
                code.writeline(raw_index_load)
                code.writeline(
                    "valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)"
                )
                coord_normalize = (
                    f"coord_{i} = tl.where("
                    "raw_index < 0, raw_index + dim_size, raw_index)"
                )
                code.writeline(coord_normalize)
            code.writeline(f"out_offsets += coord_{i} * stride_{i}")

        code.newline()
        code.writeline(
            'tl.device_assert(valid_index, "index out of bounds", mask=mask)'
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
    code.newline()
    return code


def generate_contiguous_index_fill_kernel(
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline("@libentry()")
    code.writeline("@triton.jit(debug=True)")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("out,")
        code.writeline("index,")
        code.writeline("value,")
        code.writeline("outer_index_len,")
        code.writeline("index_len,")
        code.writeline("dim_size,")
        code.writeline("inner_size,")
        code.writeline("VALUE_IS_TENSOR: tl.constexpr,")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid_m = tl.program_id(axis=0)")
        code.writeline("pid_n = tl.program_id(axis=1)")
        code.writeline("m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("inner_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
        code.writeline("m_mask = m_offsets < outer_index_len")
        code.writeline("index_coord = m_offsets % index_len")
        code.writeline("outer_coord = m_offsets // index_len")
        code.writeline(
            "raw_index = tl.load(index + index_coord, mask=m_mask, other=0).to(tl.int64)"
        )
        code.writeline(
            "valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)"
        )
        code.writeline(
            'tl.device_assert(valid_index, "index out of bounds", mask=m_mask)'
        )
        code.writeline(
            "normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)"
        )
        code.writeline("out_offsets = outer_coord[:, None] * dim_size * inner_size")
        code.writeline("out_offsets += normalized_index[:, None] * inner_size")
        code.writeline("out_offsets += inner_offsets[None, :]")
        code.writeline(
            "store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)"
        )
        code.writeline("store_mask = store_mask & valid_index[:, None]")
        code.writeline("if VALUE_IS_TENSOR:")
        with code.indent():
            code.writeline("fill_value = tl.load(value)")
        code.writeline("else:")
        with code.indent():
            code.writeline("fill_value = value")
        code.writeline("tl.store(out + out_offsets, fill_value, mask=store_mask)")

    code.newline()
    code.newline()
    return code


def generate_destination_passing_wrapper(
    rank: int,
    dim: int,
    wrapper_name: str,
    kernel_name: str,
    contiguous_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    wrapper_signature = (
        f"def {wrapper_name}("
        "out, dim, index, value, N, index_len, dim_size, value_is_tensor):"
    )
    code.writeline(wrapper_signature)
    with code.indent():
        code.writeline("out_shapes = list(out.shape)")
        code.writeline("out_strides = list(out.stride())")
        code.writeline("BLOCK_SIZE = 512")
        code.writeline("if out.is_contiguous():")
        with code.indent():
            code.writeline("inner_size = 1")
            for i in range(dim + 1, rank):
                code.writeline(f"inner_size *= out_shapes[{i}]")
            code.writeline("block_n = 1")
            code.writeline("block_m = BLOCK_SIZE")
            code.writeline("if inner_size > 1:")
            with code.indent():
                code.writeline("block_n = min(64, triton.next_power_of_2(inner_size))")
                code.writeline("if inner_size <= 4:")
                with code.indent():
                    code.writeline("block_m = BLOCK_SIZE")
                code.writeline("else:")
                with code.indent():
                    code.writeline("block_m = max(1, BLOCK_SIZE // block_n)")
            code.writeline("outer_index_len = N // inner_size")
            code.writeline("grid = (")
            with code.indent():
                code.writeline("triton.cdiv(outer_index_len, block_m),")
                code.writeline("triton.cdiv(inner_size, block_n),")
            code.writeline(")")
            code.writeline(f"{contiguous_kernel_name}[grid](")
            with code.indent():
                code.writeline("out,")
                code.writeline("index,")
                code.writeline("value,")
                code.writeline("outer_index_len,")
                code.writeline("index_len,")
                code.writeline("dim_size,")
                code.writeline("inner_size,")
                code.writeline("VALUE_IS_TENSOR=value_is_tensor,")
                code.writeline("BLOCK_M=block_m,")
                code.writeline("BLOCK_N=block_n,")
            code.writeline(")")
            code.writeline("return out")
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


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    out = inputs[0]
    dim = inputs[1]
    rank = out.ndim
    contiguous_kernel_name = "_index_fill_contiguous_jit_function"

    code = generate_imports(code)
    code = generate_index_fill_kernel(rank, dim, kernel_name, code)
    code = generate_contiguous_index_fill_kernel(contiguous_kernel_name, code)
    code = generate_destination_passing_wrapper(
        rank,
        dim,
        wrapper_name,
        kernel_name,
        contiguous_kernel_name,
        code,
    )
    return code


class IndexFillFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        if key in self.overloads:
            return self.overloads[key](*args, **kwargs)

        code = IndentedBuffer()
        code = generate_code(
            args,
            "_index_fill_wrapper",
            "_index_fill_jit_function",
            code,
        )
        file_name = f"index_fill_{key}_pid_{self.pid}.py"
        file_path = code_cache_dir() / file_name
        write_atomic(file_path, code.getvalue())

        spec = importlib.util.spec_from_file_location(
            f"_gen_index_fill_{key}_pid_{self.pid}",
            file_path,
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        overload = getattr(m, "_index_fill_wrapper")
        self.overloads[key] = overload
        return overload(*args, **kwargs)

    def arg_key(self, *args):
        out = args[0]
        dim = args[1]
        return f"rank_{out.ndim}_dim_{dim}"


_index_fill_func = IndexFillFunction()


def _prepare_index(inp, dim, index):
    if inp.ndim == 0:
        raise IndexError("index_fill expects self to have at least one dimension")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    dim = dim % inp.ndim

    if index.dtype != torch.long:
        raise IndexError("index_fill_(): Expected dtype int64 for index.")
    if index.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {index.device}!"
        )
    if index.ndim > 1:
        raise IndexError("index_fill_(): Index is supposed to be a vector")
    if index.ndim == 0:
        index = index.reshape(1)

    if index.numel() > 0:
        _check_index_bounds(inp, dim, index)

    return dim, index


def _check_index_bounds(inp, dim, index):
    mode = os.environ.get("FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK", "device").lower()
    if _index_fill_uses_device_bounds_check():
        return

    dim_size = inp.size(dim)
    if mode in ("sync", "1", "true", "on"):
        min_index = int(torch.min(index).item())
        max_index = int(torch.max(index).item())
        if min_index < -dim_size or max_index >= dim_size:
            raise IndexError("index out of range in self")
        return

    if mode == "async":
        valid = ((index >= -dim_size) & (index < dim_size)).all()
        torch.ops.aten._assert_async.msg(valid, "index out of range in self")
        return

    raise ValueError(
        "FLAG_GEMS_INDEX_FILL_BOUNDS_CHECK must be one of device, sync, or async"
    )


def _prepare_tensor_value(inp, value):
    if value.ndim != 0:
        raise RuntimeError(
            "index_fill_ only supports a 0-dimensional value tensor, "
            f"but got tensor with {value.ndim} dimension(s)."
        )
    if value.device.type == "cpu":
        return False, value.item()
    if value.device != inp.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least "
            f"two devices, {inp.device} and {value.device}!"
        )
    return True, value


def _index_fill_impl(out, dim, index, value, value_is_tensor):
    if out.numel() == 0 or index.numel() == 0:
        return out

    dim_size = out.size(dim)
    fill_numel = out.numel() // dim_size * index.numel()
    with torch_device_fn.device(out.device):
        _index_fill_func(
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


def index_fill_scalar(inp, dim, index, value):
    logger.debug("GEMS INDEX_FILL SCALAR")
    cpp_out = _try_cpp_index_fill_scalar_fast(inp, dim, index, value)
    if cpp_out is not None:
        return cpp_out
    dim, index = _prepare_index(inp, dim, index)
    out = _native_clone(inp)
    if not _should_skip_cpp_index_fill_out(out, dim):
        cpp_out = _try_cpp_index_fill_scalar_(out, dim, index, value)
        if cpp_out is not None:
            return cpp_out
    return _index_fill_impl(out, dim, index, value, False)


def index_fill_tensor(inp, dim, index, value):
    logger.debug("GEMS INDEX_FILL TENSOR")
    dim, index = _prepare_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    out = _native_clone(inp)
    return _index_fill_impl(out, dim, index, value, value_is_tensor)


def index_fill_scalar_out(inp, dim, index, value, *, out):
    logger.debug("GEMS INDEX_FILL SCALAR_OUT")
    dim, index = _prepare_index(inp, dim, index)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    _native_copy_(out, inp)
    if not _should_skip_cpp_index_fill_out(out, dim):
        cpp_out = _try_cpp_index_fill_scalar_(out, dim, index, value)
        if cpp_out is not None:
            return cpp_out
    return _index_fill_impl(out, dim, index, value, False)


def index_fill_tensor_out(inp, dim, index, value, *, out):
    logger.debug("GEMS INDEX_FILL TENSOR_OUT")
    dim, index = _prepare_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    _native_copy_(out, inp)
    return _index_fill_impl(out, dim, index, value, value_is_tensor)


def index_fill_scalar_(inp, dim, index, value):
    logger.debug("GEMS INDEX_FILL_ SCALAR")
    cpp_out = _try_cpp_index_fill_scalar_fast_(inp, dim, index, value)
    if cpp_out is not None:
        return cpp_out
    dim, index = _prepare_index(inp, dim, index)
    cpp_out = _try_cpp_index_fill_scalar_(inp, dim, index, value)
    if cpp_out is not None:
        return cpp_out
    return _index_fill_impl(inp, dim, index, value, False)


def index_fill_tensor_(inp, dim, index, value):
    logger.debug("GEMS INDEX_FILL_ TENSOR")
    dim, index = _prepare_index(inp, dim, index)
    value_is_tensor, value = _prepare_tensor_value(inp, value)
    return _index_fill_impl(inp, dim, index, value, value_is_tensor)
