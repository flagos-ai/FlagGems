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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


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


def _resolve_lazy_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    return tensor


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


def _assert_index_in_bounds(index_1d: torch.Tensor, upper_bound: int) -> None:
    if index_1d.numel() == 0:
        return
    assert bool(
        ((0 <= index_1d) & (index_1d < upper_bound)).all().item()
    ), "0 <= index < self.size(dim)"


def _launch_index_add(out, index_1d, src, dim: int, inp_shape_dim: int, alpha):
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
    logger.debug("GEMS INDEX_ADD")
    dim = _validate_index_add_args(inp, dim, index, src, inplace=False)
    inp_resolved = _resolve_lazy_view(inp)
    index_1d = _resolve_lazy_view(index).reshape(-1).contiguous()
    src_resolved = _resolve_lazy_view(src)
    inp_shape_dim = inp.size(dim) if inp.ndim > 0 else 1

    _assert_index_in_bounds(index_1d, inp_shape_dim)
    if src_resolved.numel() == 0:
        return inp_resolved.clone(memory_format=torch.contiguous_format)

    # MetaX workaround: tl.atomic_add does not support bf16 in MLIR backend.
    # Cast to float32, perform the operation, and cast back.
    orig_dtype = inp.dtype
    needs_cast = orig_dtype == torch.bfloat16
    if needs_cast:
        out = inp_resolved.to(
            dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
        )
        work_src = src_resolved.to(torch.float32)
    else:
        out = inp_resolved.clone(memory_format=torch.contiguous_format)
        work_src = src_resolved

    _launch_index_add(out, index_1d, work_src, dim, inp_shape_dim, alpha)
    if needs_cast:
        return out.to(orig_dtype)
    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS INDEX_ADD_")
    dim = _validate_index_add_args(inp, dim, index, src, inplace=True)
    index_1d = _resolve_lazy_view(index).reshape(-1).contiguous()
    src_resolved = _resolve_lazy_view(src)
    inp_shape_dim = inp.size(dim) if inp.ndim > 0 else 1

    _assert_index_in_bounds(index_1d, inp_shape_dim)
    if src_resolved.numel() == 0:
        return inp

    if (
        inp.numel() > 0
        and index_1d.numel() > 0
        and inp.untyped_storage().data_ptr() == index_1d.untyped_storage().data_ptr()
    ):
        index_1d = index_1d.clone()
    if (
        inp.numel() > 0
        and src_resolved.numel() > 0
        and inp.untyped_storage().data_ptr()
        == src_resolved.untyped_storage().data_ptr()
    ):
        src_resolved = src_resolved.clone()

    target_inp = inp
    if inp.is_neg():
        target_inp = torch._neg_view(inp)
        alpha = -alpha
    if target_inp.is_conj():
        target_inp = target_inp.resolve_conj()

    # MetaX workaround: tl.atomic_add does not support bf16 in MLIR backend.
    # Cast to float32, perform the operation, and cast back in-place.
    orig_dtype = target_inp.dtype
    needs_cast = orig_dtype == torch.bfloat16
    if needs_cast:
        out = target_inp.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        work_src = src_resolved.to(torch.float32)
    else:
        out = target_inp
        work_src = src_resolved

    _launch_index_add(out, index_1d, work_src, dim, inp_shape_dim, alpha)
    if needs_cast or out is not target_inp:
        target_inp.copy_(out.to(orig_dtype) if needs_cast else out)
    return inp
