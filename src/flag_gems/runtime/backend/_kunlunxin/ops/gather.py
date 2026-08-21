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

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

from .nonzero import _device_int_tensor

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("import builtins")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")

    code.newline()
    code.newline()
    return code


# ---------------------------------------------------------------------------
# Fast path (int32-safe sizes, index.numel() < 2**31):
#
# The index tensor is contiguous and flattened to (M, N) with
# N = index.shape[-1].  For element (m, j) of that flat view (j = the last
# axis coordinate, m = the coordinates of the remaining leading axes):
#
#   out_addr(m, j) = idx_addr(m, j) = m * N + j          (contiguous DMA)
#   inp_addr(m, j) = base(m) + j * stride_last + idx(m, j) * stride_dim
#
# where base(m) = sum_{i < rank-1, i != dim} digit_i(m) * inp_stride_i and
# digit_i(m) are the mixed-radix digits of m with radices index.shape[0..r-2],
# and stride_last = inp.stride(rank-1) (0 when dim == rank-1 so the j term
# vanishes -- the 'dim' axis is driven entirely by idx * stride_dim).
#
# The per-row base is a [BLOCK_M] int32 vector computed once per program
# (rank-1 constexpr div/mod ops), and every per-element address is a plain
# int32 multiply-add -- no per-element int64 div/mod chains, no int64 offset
# tile materialization.  index/out traffic is contiguous block DMA (stride-1);
# only the data-dependent inp load remains a discrete gather (XPU structural
# ceiling).
# ---------------------------------------------------------------------------


def generate_gather_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("inp,")
        code.writeline("out,")
        code.writeline("index,")
        code.writeline("dim: tl.constexpr,")
        code.writeline("stride_dim: tl.constexpr,")
        code.writeline("stride_last: tl.constexpr,")
        if rank > 1:
            stride_args = ", ".join(
                f"index_shape_{i}: tl.constexpr" for i in range(rank - 1)
            )
            code.writeline(f"{stride_args}, # shape of the leading axes (index)")
            stride_args = ", ".join(
                f"inp_stride_{i}: tl.constexpr" for i in range(rank - 1)
            )
            code.writeline(f"{stride_args}, # stride for inp leading axes")
        code.writeline("M: tl.constexpr,")
        code.writeline("N: tl.constexpr,")
        code.writeline("BLOCK_M: tl.constexpr,")
        code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid_m = ext.program_id(0)")
        code.writeline("pid_n = ext.program_id(1)")
        code.writeline("rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        code.writeline("cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")

        # per-row base over the leading axes 0..rank-2 (excluding the gather dim)
        if rank > 1:
            code.writeline("cur = rows")
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int32)")
            for i in range(rank - 2, -1, -1):
                code.writeline(f"acc = cur % index_shape_{i}")
                code.writeline(
                    f"base += tl.where(dim == {i}, 0, acc) * inp_stride_{i}"
                )
                code.writeline(f"cur //= index_shape_{i}")

        code.writeline("offsets = rows[:, None] * N + cols[None, :]")
        if rank == 1:
            code.writeline("base = tl.zeros((BLOCK_M,), dtype=tl.int32)")

        # Always-masked path: the XPU backend produced illegal-access crashes
        # for unmasked small-tile gathers and PassManager failures for some
        # narrow masked variants; the always-masked formulation is proven
        # stable across the tiny/mid-size matrix and costs ~1% on big tiles.
        code.writeline("mask = (rows < M)[:, None] & (cols < N)[None, :]")
        code.writeline("cur_index = tl.load(index + offsets, mask=mask, other=0)")
        code.writeline("inp_offsets = base[:, None] + cur_index.to(tl.int32) * stride_dim")
        if rank > 1:
            code.writeline("inp_offsets += cols[None, :] * stride_last")
        code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
        code.writeline("tl.store(out + offsets, cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # inp, out, index, dim, stride_dim, inp_dim_size, M, N
    parameters: List[str] = []

    parameters.append("inp")
    parameters.append("out")
    parameters.append("index")
    parameters.append("dim")
    parameters.append("stride_dim")
    parameters.append("stride_last")
    parameters.append("M")
    parameters.append("N")

    return ", ".join(parameters)


def generate_gather_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_strides = list(inp.stride())")

        # Bounded tile: BLOCK_N up to 4096, BLOCK_M mirrors the proven
        # cdiv(M, 12) heuristic (cap 8).  Keeping BLOCK_M == 1 for small M is
        # mandatory: wider [BM, BN] tiles with BM ~ 2-4 and narrow BN trigger
        # XPU backend illegal-access / PassManager failures on tiny shapes.
        code.writeline("BLOCK_N = builtins.min(triton.next_power_of_2(N), 4096)")
        code.writeline(
            "BLOCK_M = builtins.min(triton.next_power_of_2(triton.cdiv(M, 12)), 8)"
        )

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(M, meta["BLOCK_M"]),')
            code.writeline('triton.cdiv(N, meta["BLOCK_N"])')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("inp, out, index, ")
            code.writeline("dim,")
            code.writeline("stride_dim,")
            code.writeline("stride_last,")
            if rank > 1:
                s = ", ".join(f"index_shapes[{i}]" for i in range(rank - 1))
                code.writeline(f"{s},")
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank - 1))
                code.writeline(f"{s},")
            code.writeline("M,")
            code.writeline("N,")
            code.writeline("BLOCK_M=BLOCK_M,")
            code.writeline("BLOCK_N=BLOCK_N,")
            code.writeline("buffer_size_limit=2048,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: inp, out, index, dim, stride_dim, stride_last, M, N
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_gather_kernel(rank, kernel_name, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
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
                "_gather_wrapper",
                "_gather_jit_function",
                code,
            )

            file_name = f"gather_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_gather_func = GatherFunction()


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_KUNLUNXIN GATHER")
    if dim < 0:
        dim += inp.ndim
    if inp.ndim != index.ndim:
        raise IndexError(
            f"Index tensor must have the same number of dimensions as input tensor. "
            f"Got {index.ndim} and {inp.ndim}."
        )

    inp = inp.contiguous()
    index = index.contiguous()
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    out = out.contiguous()
    if index.numel() == 0:
        return out

    stride_dim = inp.stride(dim)
    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N
    inp_dim_size = inp.size(dim)

    if inp.numel() < 2**31 and index.numel() < 2**31:
        stride_last = inp.stride(index.ndim - 1) if dim != index.ndim - 1 else 0
        _gather_func(inp, out, index, dim, stride_dim, stride_last, M, N)
    else:
        # int64 fallback path for gigantic (>= 2^31 elements) tensors.
        inp_strided = restride_dim(inp, dim, index.shape)
        _gather_func_legacy(
            inp_strided, out, index, dim, stride_dim, inp_dim_size, M, N
        )
    return out


# ---------------------------------------------------------------------------
# Legacy generated kernel (int64 offsets, unbounded shapes).  Kept only as the
# fallback for index/input sizes >= 2^31 where the int32 fast path could
# overflow.  Never used by the standard test/benchmark matrix.
# ---------------------------------------------------------------------------


def generate_gather_legacy_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    code.writeline("def heur_block_m(args):")
    with code.indent():
        code.writeline(
            'return builtins.min(triton.next_power_of_2(triton.cdiv(args["M"], 12)), 8)'
        )

    code.newline()

    code.writeline("def heur_block_n(args):")
    with code.indent():
        code.writeline('return builtins.min(triton.next_power_of_2(args["N"]), 4096)')

    code.newline()
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("values={")
        with code.indent():
            code.writeline('"BLOCK_M": heur_block_m,')
            code.writeline('"BLOCK_N": heur_block_n,')
        code.writeline("},")
    code.writeline(")")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("inp,")
            code.writeline("out,")
            code.writeline("index,")

            stride_args = ", ".join(
                f"inp_stride_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(
                f"index_stride_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(
                f"index_shape_{i}: tl.constexpr" for i in range(rank)
            )
            code.writeline(f"{shape_args}, # shape for index")

            code.writeline("dim: tl.constexpr,")
            code.writeline("stride_dim: tl.constexpr,")
            code.writeline("inp_dim_size: tl.constexpr,")
            code.writeline("M: tl.constexpr,")
            code.writeline("N: tl.constexpr,")
            code.writeline("BLOCK_M: tl.constexpr,")
            code.writeline("BLOCK_N: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid_x = ext.program_id(0)")
        code.writeline("pid_y = ext.program_id(1)")
        code.writeline(
            "rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]"
        )
        code.writeline(
            "cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]"
        )
        code.writeline("rows_mask = rows_offsets < M")
        code.writeline("cols_mask = cols_offsets < N")

        code.writeline("offsets = (rows_offsets * N + cols_offsets).to(tl.int64)")
        code.writeline("mask = rows_mask & cols_mask")

        code.writeline("inp_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("idx_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("cur_idx = rows_offsets * N + cols_offsets")

        for i in range(rank):
            code.writeline(f"mod = cur_idx % index_shape_{i}")
            code.writeline(
                f"inp_offsets += tl.where(dim == {i}, 0, mod) * inp_stride_{i}"
            )
            code.writeline(f"idx_offsets += mod * index_stride_{i}")
            if i != (rank - 1):
                code.writeline(f"cur_idx //= index_shape_{i}")

        code.writeline("cur_index = tl.load(index + idx_offsets, mask=mask, other=0)")
        code.writeline("inp_offsets += cur_index * stride_dim")
        code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
        code.writeline("tl.store(out + idx_offsets, cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_legacy_wrapper() -> str:
    parameters: List[str] = []

    parameters.append("inp_strided")
    parameters.append("out")
    parameters.append("index")
    parameters.append("dim")
    parameters.append("stride_dim")
    parameters.append("inp_dim_size")
    parameters.append("M")
    parameters.append("N")

    return ", ".join(parameters)


def generate_gather_legacy_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_legacy_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = inp_strided.stride()")
        code.writeline("index_strides = index.stride()")
        code.writeline("index_shapes = list(index.shape)")

        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(M, meta["BLOCK_M"]),')
            code.writeline('triton.cdiv(N, meta["BLOCK_N"])')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("inp_strided, out, index, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("dim,")
                code.writeline("stride_dim,")
                code.writeline("inp_dim_size,")
                code.writeline("M,")
                code.writeline("N,")
                code.writeline("buffer_size_limit=2048,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_legacy_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_gather_legacy_kernel(rank, kernel_name, code)
    code = generate_gather_legacy_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunctionLegacy:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_legacy_code(
                args,
                "_gather_legacy_wrapper",
                "_gather_legacy_jit_function",
                code,
            )

            file_name = f"gather_legacy_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_legacy_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_legacy_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_gather_func_legacy = GatherFunctionLegacy()


@triton.jit
def _gather_backward_kernel(
    grad,
    index,
    output,
    self_shape,
    index_shape,
    index_strides,
    total,
    index_dim_size,
    dim: tl.constexpr,
    ndim: tl.constexpr,
    BLOCK_OUTPUT: tl.constexpr,
    BLOCK_INDEX: tl.constexpr,
):
    output_offsets_flat = tl.program_id(0) * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    output_valid_flat = output_offsets_flat < total

    remaining = output_offsets_flat
    index_base = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int64)
    output_dim_index = tl.zeros((BLOCK_OUTPUT,), dtype=tl.int64)
    coordinate_valid = output_valid_flat
    for axis in tl.static_range(ndim - 1, -1, -1):
        axis_size = tl.load(self_shape + axis)
        coordinate = remaining % axis_size
        remaining //= axis_size
        if axis == dim:
            output_dim_index = coordinate
        else:
            index_axis_size = tl.load(index_shape + axis)
            index_axis_stride = tl.load(index_strides + axis)
            coordinate_valid &= coordinate < index_axis_size
            index_base += coordinate * index_axis_stride

    index_dim_offsets = tl.arange(0, BLOCK_INDEX)[None, :]
    index_dim_stride = tl.load(index_strides + dim)
    valid = coordinate_valid[:, None] & (index_dim_offsets < index_dim_size)
    gather_offsets = index_base[:, None] + index_dim_offsets * index_dim_stride
    gathered_indices = tl.load(index + gather_offsets, mask=valid, other=-1)
    grad_values = tl.load(grad + gather_offsets, mask=valid, other=0.0)
    grad_values = tl.where(
        valid & (gathered_indices == output_dim_index[:, None]),
        grad_values.to(tl.float32),
        0.0,
    )
    result = tl.sum(grad_values, axis=1)
    result = tl.where(coordinate_valid, result, 0.0)
    tl.store(output + output_offsets_flat, result, mask=output_valid_flat)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_KUNLUNXIN GATHER_BACKWARD")
    if sparse_grad:
        raise RuntimeError("gather_backward with sparse_grad=True is not supported")

    result = grad.new_zeros(self.shape)
    if result.numel() == 0:
        return result

    dim = dim % self.ndim
    index_contiguous = index.contiguous()
    self_shape = _device_int_tensor(self.shape, torch.int64, self.device)
    index_shape = _device_int_tensor(index_contiguous.shape, torch.int64, self.device)
    index_strides = _device_int_tensor(
        index_contiguous.stride(), torch.int64, self.device
    )
    index_dim_size = index_contiguous.shape[dim]
    block_index = triton.next_power_of_2(index_dim_size)
    block_output = min(64, max(1, 2048 // block_index))
    _gather_backward_kernel[(triton.cdiv(result.numel(), block_output),)](
        grad.contiguous(),
        index_contiguous,
        result,
        self_shape,
        index_shape,
        index_strides,
        result.numel(),
        index_dim_size,
        dim=dim,
        ndim=self.ndim,
        BLOCK_OUTPUT=block_output,
        BLOCK_INDEX=block_index,
        num_warps=1,
        buffer_size_limit=2048,
        isCloseVectorization=True,
    )
    return result