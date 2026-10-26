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
import math
import operator
import os
from numbers import Number, Real
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)
_SCALAR_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_add_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("inp_numel,")
            code.writeline("inp_stride_dim,")
            code.writeline("inp_shape_dim,")
            code.writeline("src_shape_dim,")
            code.writeline("delta,")
            code.writeline("alpha,")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"src_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")
            code.writeline("USE_INT64_SOURCE: tl.constexpr,")

        code.writeline("):")

        # Kernel Code
        with code.indent():
            code.writeline("pid = tl.program_id(axis=0)")
            code.writeline("if USE_INT64_SOURCE:")
            with code.indent():
                code.writeline("pid = pid.to(tl.int64)")
            code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < N")

            for i in range(rank - 1, -1, -1):
                code.writeline(f"src_offset{i} = offsets % src_shape_{i}")
                code.writeline(f"offsets = offsets // src_shape_{i}")
            code.newline()
            comp = [f"src_offset{i} * src_stride_{i}" for i in range(rank)]
            code.writeline(f"src_offset = {' + '.join(comp)}")

            code.writeline("if USE_INT64_SOURCE:")
            with code.indent():
                code.writeline(
                    "pre_cal = tl.cast(inp_stride_dim, tl.int64) * src_shape_dim"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline("pre_cal = inp_stride_dim * src_shape_dim")

            # index add
            code.writeline("pre_idx = (src_offset // pre_cal).to(tl.int64)")
            code.writeline(
                "dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)"
            )
            code.writeline(
                "src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)"
            )
            code.writeline(
                "tl.device_assert((~mask) | ((src_dim_idx >= 0) & "
                '(src_dim_idx < inp_shape_dim)), "0 <= index < self.size(dim)")'
            )
            code.writeline(
                "input_idx = (src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)"
            )

            code.writeline(
                "input_mask = mask & (input_idx >= 0) & (input_idx < inp_numel)"
            )
            code.writeline(
                "add_on = tl.load(src + src_offset, mask=mask, other=0) * alpha"
            )
            code.writeline(
                "tl.atomic_add(out + input_idx, add_on, mask=input_mask, sem='relaxed')"
            )
            # TODO: tl.atomic_add doesn't support bfloat16! The following method may be unsafe.
            # code.writeline("cur_out = tl.load(out + input_idx, mask=input_mask)")
            # code.writeline("tl.store(out + input_idx, cur_out + add_on, mask=input_mask)")

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, src, dim, inp_stride_dim, src_shape_dim, delta, N, inp.numel(), alpha
    parameters: List[str] = []
    parameters.append("out")
    parameters.append("index")
    parameters.append("src")
    parameters.append("dim")
    parameters.append("inp_stride_dim")
    parameters.append("inp_shape_dim")
    parameters.append("src_shape_dim")
    parameters.append("delta")
    parameters.append("N")
    parameters.append("inp_numel")
    parameters.append("alpha")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name} ({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")

        # kernel launch
        code.writeline("BLOCK_SIZE = 128")  # BLOCK_SIZE setting
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, src, out, N, inp_numel, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, alpha, "
            )
            if rank > 0:
                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE,")
            # Pointer lowering on some backends scales byte offsets in the
            # source index type. Preserve cheap int32 division where it fits.
            code.writeline(
                "USE_INT64_SOURCE=src.numel() * src.element_size() > 2147483647"
            )
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, N, inp.numel(), alpha]
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_index_add_kernel(rank, kernel_name, code)
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

            # load
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
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_index_add_func = IndexAddFunction()


def _normalize_dim(inp, dim):
    dim = operator.index(dim)
    rank = max(inp.ndim, 1)
    if not -rank <= dim < rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-rank}, {rank - 1}], but got {dim})"
        )
    return dim % rank


def _validate_index_add_args(inp, dim, index, src):
    dim = _normalize_dim(inp, dim)
    if index.ndim > 1:
        raise IndexError("Index is supposed to be a vector")
    if index.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("Expected dtype int32/int64 for index")
    if inp.dtype != src.dtype:
        raise RuntimeError("Self and source must have the same scalar type")
    if inp.device != src.device or index.device != inp.device:
        raise RuntimeError("Expected all tensors to be on the same device")
    if inp.ndim != src.ndim or any(
        inp.size(i) != src.size(i) for i in range(inp.ndim) if i != dim
    ):
        raise RuntimeError(
            "source tensor shape must match self tensor shape, "
            "excluding the specified dimension"
        )
    src_size = src.size(dim) if src.ndim else 1
    if index.numel() != src_size:
        raise RuntimeError("Number of indices should be equal to source.size(dim)")
    return dim


def _resolve_kernel_tensor(tensor):
    # Pointer kernels access physical storage, not the lazy-negative value.
    if tensor.is_neg():
        tensor = torch.neg(torch._neg_view(tensor))
    return tensor.contiguous()


def _normalize_index_add_alpha(inp, alpha):
    # The usual identity scale needs no allocation. Other scales must pass
    # ATen's scalar conversion even when there is no source work to launch.
    if type(alpha) in (int, float) and alpha in (-1, 0, 1):
        return alpha
    if isinstance(alpha, torch.Tensor):
        if alpha.ndim:
            raise TypeError("alpha must be a number or a scalar tensor")
        alpha = alpha.item()
    if not isinstance(alpha, Number):
        raise TypeError("alpha must be a number")
    # BF16 already accumulates in FP32. Do not round its multiplier to BF16
    # before the existing FP32 multiply (e.g. -256 + 2560 * 0.1 == 0).
    conversion_dtype = torch.float32 if inp.dtype == torch.bfloat16 else inp.dtype
    converted = torch.ops.aten.scalar_tensor.default.redispatch(
        _SCALAR_KEYSET, alpha, dtype=conversion_dtype, device=torch.device("cpu")
    ).item()
    # Tensor construction can silently round a finite Half overflow to its
    # largest value or infinity; Native index_add uses checked conversion.
    # Explicit infinity/NaN remain legal floating scales.
    real_alpha = alpha.real
    if (
        inp.is_floating_point()
        and isinstance(real_alpha, Real)
        and math.isfinite(real_alpha)
        and abs(real_alpha) > torch.finfo(inp.dtype).max
    ):
        raise RuntimeError(f"value cannot be converted to {inp.dtype} without overflow")
    return converted


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() and not bool(((index >= 0) & (index < upper_bound)).all().item()):
        raise AssertionError("0 <= index < self.size(dim)")


def _assert_no_overlap(inp, other):
    # ATen rejects identical views, including an identical empty tensor.
    if inp is other:
        raise RuntimeError("input overlaps with source or index; clone it first")
    if not inp.numel() or not other.numel() or not torch._C._is_alias_of(inp, other):
        return
    # Dense views occupy a single byte interval. Sharing storage alone is not
    # overlap: disjoint slices are legal. For non-dense views ATen likewise
    # permits indeterminate overlap; materialization below prevents read races.
    if torch.ops.aten.is_non_overlapping_and_dense(
        inp
    ) and torch.ops.aten.is_non_overlapping_and_dense(other):
        start = inp.data_ptr()
        other_start = other.data_ptr()
        if (
            start < other_start + other.numel() * other.element_size()
            and other_start < start + inp.numel() * inp.element_size()
        ):
            raise RuntimeError("input overlaps with source or index; clone it first")


def _launch_index_add(out, dim, index, src, alpha):
    # The existing address decomposition assumes contiguous source/output.
    # Scalars use the same rank-one kernel without changing the public shape.
    work_out = out.reshape(1) if out.ndim == 0 else out
    src = src.reshape(1) if src.ndim == 0 else src
    with torch_device_fn.device(out.device):
        _index_add_func(
            work_out,
            index,
            src,
            dim,
            # Singleton dimensions can retain arbitrary strides even after
            # contiguous(); addressing needs the logical trailing width.
            math.prod(work_out.shape[dim + 1 :]),
            work_out.size(dim),
            src.size(dim),
            work_out.size(dim) - src.size(dim),
            src.numel(),
            work_out.numel(),
            alpha,
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX_ADD")
    dim = _validate_index_add_args(inp, dim, index, src)
    alpha = _normalize_index_add_alpha(inp, alpha)
    if src.numel() == 0:
        return inp.clone(memory_format=torch.contiguous_format)

    index = _resolve_kernel_tensor(index).reshape(-1)
    _assert_index_in_bounds(index, inp.size(dim) if inp.ndim else 1)
    work_inp = _resolve_kernel_tensor(inp)
    work_src = _resolve_kernel_tensor(src)
    needs_cast = inp.dtype == torch.bfloat16
    out = work_inp.float() if needs_cast else work_inp.clone()
    if needs_cast:
        work_src = work_src.float()
    _launch_index_add(out, dim, index, work_src, alpha)
    return out.to(inp.dtype) if needs_cast else out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX_ADD_")
    dim = _validate_index_add_args(inp, dim, index, src)
    alpha = _normalize_index_add_alpha(inp, alpha)
    _assert_no_overlap(inp, src)
    _assert_no_overlap(inp, index)
    if src.numel() == 0:
        return inp

    index = _resolve_kernel_tensor(index).reshape(-1)
    _assert_index_in_bounds(index, inp.size(dim) if inp.ndim else 1)
    work_inp = _resolve_kernel_tensor(inp)
    work_src = _resolve_kernel_tensor(src)
    needs_cast = inp.dtype == torch.bfloat16
    out = work_inp.float() if needs_cast else work_inp
    if needs_cast:
        work_src = work_src.float()
    _launch_index_add(out, dim, index, work_src, alpha)
    if out is not inp:
        inp.copy_(out.to(inp.dtype) if needs_cast else out)
    return inp
