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

import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_TRITON_SUPPORTS_BF16_ATOMIC_ADD = _triton_version_at_least(3, 4)
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_add_kernel(
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
        code.writeline("src,")
        code.writeline("out,")
        code.writeline("N,")
        code.writeline("inp_shape_dim,")
        code.writeline("alpha,")
        code.writeline("index_stride_0: int,")

        src_stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
        code.writeline(f"{src_stride_args},")

        src_shape_args = ", ".join(f"src_shape_{i}: int" for i in range(rank))
        code.writeline(f"{src_shape_args},")

        out_stride_args = ", ".join(f"out_stride_{i}: int" for i in range(rank))
        code.writeline(f"{out_stride_args},")

        code.writeline("BLOCK_SIZE: tl.constexpr,")
        code.writeline("INDEX_64: tl.constexpr,")

    code.writeline("):")

    with code.indent():
        code.writeline("pid = tl.program_id(axis=0)")
        code.writeline("if INDEX_64:")
        with code.indent():
            code.writeline(
                "offsets = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)"
            )
        code.writeline("else:")
        with code.indent():
            code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < N")

        code.writeline("rem = offsets")
        for i in range(rank - 1, -1, -1):
            code.writeline(f"src_offset{i} = rem % src_shape_{i}")
            if i > 0:
                code.writeline(f"rem = rem // src_shape_{i}")
        code.newline()

        src_comp = [f"src_offset{i}.to(tl.int64) * src_stride_{i}" for i in range(rank)]
        code.writeline(f"src_offset = {' + '.join(src_comp)}")

        code.writeline(
            f"src_dim_idx = tl.load(index + src_offset{dim}.to(tl.int64) * index_stride_0, "
            "mask=mask, other=0).to(tl.int64)"
        )
        code.writeline(
            "valid = mask & (src_dim_idx >= 0) & (src_dim_idx < inp_shape_dim)"
        )

        out_comp = [
            (
                f"src_dim_idx * out_stride_{i}"
                if i == dim
                else f"src_offset{i}.to(tl.int64) * out_stride_{i}"
            )
            for i in range(rank)
        ]
        code.writeline(f"out_offset = {' + '.join(out_comp)}")

        code.writeline(
            "add_on = (tl.load(src + src_offset, mask=mask, other=0) * alpha).to(out.dtype.element_ty)"
        )
        code.writeline(
            "tl.atomic_add(out + out_offset, add_on, mask=valid, sem='relaxed')"
        )

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = [
        "out",
        "index",
        "src",
        "dim",
        "inp_shape_dim",
        "N",
        "alpha",
    ]
    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    code.writeline(f"def {wrapper_name}({parameters}):")

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")
        code.writeline("out_strides = list(out.stride())")
        code.writeline("index_stride_0 = index.stride(0) if index.ndim > 0 else 1")

        code.writeline("BLOCK_SIZE = 128")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            code.writeline("index, src, out, N, inp_shape_dim, alpha, index_stride_0,")
            s = ", ".join(f"src_strides[{i}]" for i in range(rank))
            code.writeline(f"{s},")
            s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
            code.writeline(f"{s},")
            s = ", ".join(f"out_strides[{i}]" for i in range(rank))
            code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE,")
            code.writeline("INDEX_64=(N >= (1 << 31)),")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any, ...],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: (out, index, src, dim, inp_shape_dim, N, alpha)
    rank = len(inputs[2].shape)
    dim = int(inputs[3])

    code = generate_imports(code)
    code = generate_index_add_kernel(rank, dim, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexAddFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_add_wrapper",
                "_index_add_jit_function",
                code,
            )

            file_name = f"index_add_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_add_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        rank = args[2].ndim
        dim = int(args[3])
        return f"{rank}_{dim}"


_index_add_func = IndexAddFunction()


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _has_lazy_view(tensor: torch.Tensor) -> bool:
    return tensor.is_neg() or tensor.is_conj()


def _resolve_lazy_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    return tensor


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.numel() > 0
        and b.numel() > 0
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


def _validate_index_add_args(inp, dim, index, src, inplace: bool = False) -> int:
    rank = max(inp.ndim, 1)
    assert -rank <= dim < rank, "Invalid dim"
    dim %= rank
    assert index.ndim <= 1, "Index is supposed to be a vector"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"
    src_dim_size = src.size(dim) if src.ndim > 0 else 1
    assert (
        index.numel() == src_dim_size
    ), "The dimth dimension of source must have the same size as the length of index"

    if inplace:
        if inp is src or inp is index:
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and "
                "the written-to tensor refer to a single memory location. "
                "Please clone() the tensor before performing the operation."
            )
        if any(s > 1 and st == 0 for s, st in zip(inp.shape, inp.stride())):
            raise RuntimeError(
                "unsupported operation: more than one element of the written-to "
                "tensor refers to a single memory location. Please clone() the "
                "tensor before performing the operation."
            )
    return dim


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() == 0:
        return index.reshape(-1)
    # Materialize a contiguous 1D view first so aminmax never reads strided storage holes.
    index_1d = index.reshape(-1).contiguous()
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _FALLBACK_KEYSET, index_1d, dim=None, keepdim=False
    )
    assert (
        lower.item() >= 0 and upper.item() < upper_bound
    ), "0 <= index < self.size(dim)"
    return index_1d


def _can_use_contiguous_suffix_path(inp, dim, index, src, inplace: bool = False):
    if src.numel() == 0:
        return False
    if _has_lazy_view(inp) or _has_lazy_view(index) or _has_lazy_view(src):
        return False
    if inplace and (
        inp is src
        or inp is index
        or _shares_storage(inp, src)
        or _shares_storage(inp, index)
    ):
        return False
    if not (
        inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False

    suffix_size = _volume(src.shape[dim + 1 :])
    return suffix_size > 1


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    offsets = ext.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count

    cols = offsets % suffix_size
    rows = offsets // suffix_size
    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(
        out + out_offsets,
        (values * alpha).to(out.dtype.element_ty),
        mask=valid,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_fp16_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_fp16_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count

    cols = offsets % suffix_size
    rows = offsets // suffix_size
    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(
        out + out_offsets,
        (values * alpha).to(out.dtype.element_ty),
        mask=valid,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_tile"),
    key=["row_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_tile_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    col_mask = cols < suffix_size
    mask = row_mask & col_mask

    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=row_mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(
        out + out_offsets,
        (values * alpha).to(out.dtype.element_ty),
        mask=valid,
        sem="relaxed",
    )


def _run_contiguous_suffix_flat_path(
    out, dim, index, src, alpha, use_fp16_config=False
):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    total_count = row_count * suffix_size
    grid = lambda meta: (triton.cdiv(total_count, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        if use_fp16_config:
            _index_add_contiguous_suffix_fp16_flat_kernel[grid](
                out,
                index,
                src,
                total_count,
                index.numel(),
                out.size(dim),
                suffix_size,
                alpha,
            )
        else:
            _index_add_contiguous_suffix_flat_kernel[grid](
                out,
                index,
                src,
                total_count,
                index.numel(),
                out.size(dim),
                suffix_size,
                alpha,
                ACCUMULATE_FP32=(
                    out.dtype == torch.float32 and src.dtype == torch.bfloat16
                ),
            )
    return out


def _run_contiguous_suffix_tile_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    grid = lambda meta: (
        triton.cdiv(row_count, meta["BLOCK_M"]),
        triton.cdiv(suffix_size, meta["BLOCK_N"]),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_tile_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ACCUMULATE_FP32=(
                out.dtype == torch.float32 and src.dtype == torch.bfloat16
            ),
        )
    return out


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    # View contiguous tensors as [prefix, index_len, suffix] and scatter-add
    # dense suffix tiles without generic rank/stride address decomposition.
    suffix_size = _volume(src.shape[dim + 1 :])
    if src.dtype == torch.float16 and suffix_size <= 512:
        return _run_contiguous_suffix_flat_path(
            out, dim, index, src, alpha, use_fp16_config=True
        )
    if suffix_size <= 64:
        return _run_contiguous_suffix_flat_path(out, dim, index, src, alpha)
    return _run_contiguous_suffix_tile_path(out, dim, index, src, alpha)


def _launch_strided_index_add(out, index_1d, src, dim: int, inp_shape_dim: int, alpha):
    work_out = out.reshape(1) if out.ndim == 0 else out
    work_src = src.reshape(1) if src.ndim == 0 else src
    work_dim = 0 if out.ndim == 0 else dim
    with torch_device_fn.device(out.device):
        _index_add_func(
            work_out,
            index_1d,
            work_src,
            work_dim,
            inp_shape_dim,
            work_src.numel(),
            alpha,
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_HYGON INDEX_ADD")
    normalized_dim = (
        dim % inp.ndim if inp.ndim > 0 and -inp.ndim <= dim < inp.ndim else dim
    )
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src, inplace=False):
        index_contig = _assert_index_in_bounds(index, inp.size(normalized_dim))
        accumulate_fp32 = (
            inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
        )
        out = inp.float() if accumulate_fp32 else inp.clone()
        res = _run_contiguous_suffix_path(out, normalized_dim, index_contig, src, alpha)
        if res is not None:
            return res.to(inp.dtype) if accumulate_fp32 else res

    dim = _validate_index_add_args(inp, dim, index, src, inplace=False)
    inp_resolved = _resolve_lazy_view(inp)
    index_resolved = _resolve_lazy_view(index)
    src_resolved = _resolve_lazy_view(src)
    inp_shape_dim = inp.size(dim) if inp.ndim > 0 else 1

    index_1d = _assert_index_in_bounds(index_resolved, inp_shape_dim)
    if src_resolved.numel() == 0:
        return inp_resolved.clone(memory_format=torch.contiguous_format)

    accumulate_fp32 = (
        inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    if accumulate_fp32:
        out = inp_resolved.to(
            dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
        )
        src_for_kernel = src_resolved.to(torch.float32)
    else:
        out = inp_resolved.clone(memory_format=torch.contiguous_format)
        src_for_kernel = src_resolved

    _launch_strided_index_add(out, index_1d, src_for_kernel, dim, inp_shape_dim, alpha)
    return out.to(inp.dtype) if accumulate_fp32 else out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_HYGON INDEX_ADD_")
    normalized_dim = (
        dim % inp.ndim if inp.ndim > 0 and -inp.ndim <= dim < inp.ndim else dim
    )
    if _can_use_contiguous_suffix_path(inp, normalized_dim, index, src, inplace=True):
        index_contig = _assert_index_in_bounds(index, inp.size(normalized_dim))
        accumulate_fp32 = (
            inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
        )
        out = inp.float() if accumulate_fp32 else inp
        res = _run_contiguous_suffix_path(out, normalized_dim, index_contig, src, alpha)
        if res is not None:
            if accumulate_fp32:
                inp.copy_(res)
            return inp

    dim = _validate_index_add_args(inp, dim, index, src, inplace=True)
    index_resolved = _resolve_lazy_view(index)
    src_resolved = _resolve_lazy_view(src)
    inp_shape_dim = inp.size(dim) if inp.ndim > 0 else 1

    index_1d = _assert_index_in_bounds(index_resolved, inp_shape_dim)
    if src_resolved.numel() == 0:
        return inp

    if _shares_storage(inp, index_1d):
        index_1d = index_1d.clone()
    if _shares_storage(inp, src_resolved):
        src_resolved = src_resolved.clone()

    target_inp = inp
    if inp.is_neg():
        target_inp = torch._neg_view(inp)
        alpha = -alpha
    if target_inp.is_conj():
        target_inp = target_inp.resolve_conj()

    accumulate_fp32 = (
        target_inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    if accumulate_fp32:
        out = target_inp.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        src_for_kernel = src_resolved.to(torch.float32)
    else:
        out = target_inp
        src_for_kernel = src_resolved

    _launch_strided_index_add(out, index_1d, src_for_kernel, dim, inp_shape_dim, alpha)
    if accumulate_fp32 or out is not target_inp:
        target_inp.copy_(out.to(target_inp.dtype) if accumulate_fp32 else out)
    return inp
