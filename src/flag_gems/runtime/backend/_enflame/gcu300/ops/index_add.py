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
    code.writeline("import torch")
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
    # GCU300 cannot legalize `tt.atomic_rmw`, so the scatter has to be
    # inverted into a segmented sum: for every output row `i` along `dim`,
    # binary-search the sorted index for the contiguous run that maps
    # to `i`, then accumulate `other[order[j]]` over that run.  `index_sorted`
    # and `order` come from a `torch.sort(index)` in the wrapper.
    other_dims = [k for k in range(rank) if k != dim]

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("index_sorted,")
        code.writeline("order,")
        code.writeline("src,")
        code.writeline("out,")
        code.writeline("M,")
        code.writeline("L,")
        code.writeline("ROWS,")
        code.writeline("out_stride_dim,")
        code.writeline("src_stride_dim,")
        code.writeline("alpha,")

        for k in other_dims:
            code.writeline(f"src_stride_{k}: tl.constexpr,")
        for k in other_dims:
            code.writeline(f"out_stride_{k}: tl.constexpr,")
        for k in other_dims:
            code.writeline(f"shape_{k}: tl.constexpr,")

        code.writeline("INNER: tl.constexpr,")
        code.writeline("BLOCK: tl.constexpr,")
        code.writeline("):")

        # Kernel Code
        with code.indent():
            code.writeline("pid_r = tl.program_id(0)")
            code.writeline("pid_t = tl.program_id(1)")
            code.writeline("num_r = tl.num_programs(0)")
            code.writeline("num_t = tl.num_programs(1)")
            code.writeline("n_tiles = tl.cdiv(L, BLOCK)")
            code.newline()

            code.writeline("for row in tl.range(pid_r, ROWS, num_r):")
            with code.indent():
                # start = lower_bound(index_sorted, row)
                code.writeline("lo = 0")
                code.writeline("hi = M")
                code.writeline("while lo < hi:")
                with code.indent():
                    code.writeline("mid = lo + (hi - lo) // 2")
                    code.writeline("val = tl.load(index_sorted + mid)")
                    code.writeline("lo = tl.where(val < row, mid + 1, lo)")
                    code.writeline("hi = tl.where(val < row, hi, mid)")
                code.writeline("start = lo")
                # end = lower_bound(index_sorted, row + 1) = upper_bound(row)
                code.writeline("lo = start")
                code.writeline("hi = M")
                code.writeline("while lo < hi:")
                with code.indent():
                    code.writeline("mid = lo + (hi - lo) // 2")
                    code.writeline("val = tl.load(index_sorted + mid)")
                    code.writeline("lo = tl.where(val < row + 1, mid + 1, lo)")
                    code.writeline("hi = tl.where(val < row + 1, hi, mid)")
                code.writeline("end = lo")
                code.newline()

                code.writeline("for tile in tl.range(pid_t, n_tiles, num_t):")
                with code.indent():
                    code.writeline("lanes = tile * BLOCK + tl.arange(0, BLOCK)")
                    code.writeline("mask = lanes < L")
                    code.writeline("lane_outer = lanes // INNER")
                    code.writeline("lane_inner = lanes % INNER")

                    # decompose the non-`dim` coordinates of every lane
                    code.writeline("rem = lane_outer")
                    for k in range(dim - 1, -1, -1):
                        code.writeline(f"coord_{k} = rem % shape_{k}")
                        code.writeline(f"rem = rem // shape_{k}")
                    code.writeline("rem = lane_inner")
                    for k in range(rank - 1, dim, -1):
                        code.writeline(f"coord_{k} = rem % shape_{k}")
                        code.writeline(f"rem = rem // shape_{k}")

                    terms = [f"coord_{k} * src_stride_{k}" for k in other_dims]
                    if terms:
                        code.writeline(f"src_base = {' + '.join(terms)}")
                        terms = [f"coord_{k} * out_stride_{k}" for k in other_dims]
                        code.writeline(f"out_base = {' + '.join(terms)}")
                    else:
                        # rank 1: the lane axis is empty, but loads still need
                        # a block pointer so `mask` can be applied.
                        code.writeline("src_base = tl.zeros([BLOCK], dtype=tl.int32)")
                        code.writeline("out_base = tl.zeros([BLOCK], dtype=tl.int32)")
                    code.writeline("out_off = out_base + row * out_stride_dim")
                    code.newline()

                    code.writeline("acc = tl.load(out + out_off, mask=mask, other=0)")
                    code.writeline("j = start")
                    code.writeline("while j < end:")
                    with code.indent():
                        code.writeline("src_row = tl.load(order + j)")
                        # Triton promotes bf16 arithmetic to fp32, so cast the
                        # product back to `out`'s element type to keep the
                        # loop-carried accumulator type stable.
                        code.writeline(
                            "acc += (tl.load(src + src_base + "
                            "src_row * src_stride_dim, mask=mask, other=0)"
                            " * alpha).to(out.dtype.element_ty)"
                        )
                        code.writeline("j += 1")
                    code.writeline("tl.store(out + out_off, acc, mask=mask)")

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, N, inp.numel(), alpha
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
    dim: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    other_dims = [k for k in range(rank) if k != dim]
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name} ({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")
        code.writeline("out_strides = list(out.stride())")
        code.newline()

        # `torch.sort` is unusable here: torch_gcu dispatches it through
        # `at::native::sort()` into `at::_ops::sort_stable` whatever the
        # `stable` argument, and the CI runner's `topsatenSort` rejects that
        # with TOPSATEN_STATUS_BAD_PARAM. Sort on the host instead; `index` is
        # one integer per scatter row against the `src` elements each row moves.
        code.writeline("index_host = index.to('cpu').to(torch.int32)")
        code.writeline("index_sorted, order = torch.sort(index_host)")
        code.writeline("index_sorted = index_sorted.to(out.device)")
        code.writeline("order = order.to(torch.int32).to(out.device)")
        code.newline()

        code.writeline("M = index.numel()")
        code.writeline("L = src.numel() // src_shape_dim")
        code.writeline("BLOCK = min(1024, max(32, triton.next_power_of_2(L)))")
        code.writeline("ROWS = inp_shape_dim")
        code.writeline("grid = (")
        with code.indent():
            code.writeline("min(ROWS, 65535),")
            code.writeline("min(triton.cdiv(L, BLOCK), 255),")
        code.writeline(")")
        code.writeline("INNER = 1")
        for k in range(dim + 1, rank):
            code.writeline(f"INNER *= src_shapes[{k}]")
        code.newline()

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index_sorted, order, src, out, M, L, ROWS, "
                "inp_stride_dim, src_strides[dim], alpha,"
            )
            for k in other_dims:
                code.writeline(f"src_strides[{k}],")
            for k in other_dims:
                code.writeline(f"out_strides[{k}],")
            for k in other_dims:
                code.writeline(f"src_shapes[{k}],")
            code.writeline("INNER=INNER,")
            code.writeline("BLOCK=BLOCK,")
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
    dim = inputs[3] % rank

    code = generate_imports(code)
    code = generate_index_add_kernel(rank, dim, kernel_name, code)
    code = generate_destination_passing_wrapper(
        rank, dim, wrapper_name, kernel_name, code
    )
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
        # The generated kernel specialises on `dim`, so cache by (rank, dim).
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        dim = args[3] % max_rank
        return f"{max_rank}_{dim}"


_index_add_func = IndexAddFunction()


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    out = inp.clone()

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    if N > 0:
        with torch_device_fn.device(inp.device):
            _index_add_func(
                out,
                index,
                src,
                dim,
                inp_stride_dim,
                inp_shape_dim,
                src_shape_dim,
                delta,
                N,
                inp.numel(),
                alpha,
            )
    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD_")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert (
        ((inp.size(i) == src.size(i)) or i == dim) for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    if N > 0:
        with torch_device_fn.device(inp.device):
            _index_add_func(
                inp,
                index,
                src,
                dim,
                inp_stride_dim,
                inp_shape_dim,
                src_shape_dim,
                delta,
                N,
                inp.numel(),
                alpha,
            )
    return inp
