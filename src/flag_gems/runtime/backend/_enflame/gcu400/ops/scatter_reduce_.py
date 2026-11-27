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

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("import flag_gems")
    code.newline()
    code.newline()
    return code


def generate_scatter_reduce_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    # the autotune function
    code.writeline("def heur_block(args):")
    with code.indent():
        code.writeline("if(flag_gems.vendor_name in ['metax', 'iluvatar']):")
        with code.indent():
            code.writeline("return 256")
        code.writeline("return 128")
    code.newline()
    code.newline()

    code.writeline("def loop_count(args):")
    with code.indent():
        code.writeline("return 4")
    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    src_stride_vars = ",".join(f"'src_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{src_stride_vars},{shape_vars}])"
    )

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            # reduce options
            code.writeline("IS_SUM: tl.constexpr,")
            code.writeline("IS_PROD: tl.constexpr,")
            code.writeline("IS_AMAX: tl.constexpr,")
            code.writeline("IS_AMIN: tl.constexpr,")
            code.writeline("IS_MEAN: tl.constexpr,")
            code.writeline("IS_FLOAT32: tl.constexpr,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("src_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                    code.writeline(f"src_stride_{i} = src_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                code.writeline(f"src_offsets += mod * src_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            #   2. Use offsets to scatter
            code.writeline(
                "cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)"
            )
            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()

            # Sum reduction using atomic_add (float32) or CAS loop (bf16/fp16)
            code.writeline("if IS_SUM | IS_MEAN:")
            with code.indent():
                code.writeline("if IS_FLOAT32:")
                with code.indent():
                    code.writeline(
                        "tl.atomic_add(out + inp_offsets, cur_src, mask=mask)"
                    )
                code.writeline("else:")
                with code.indent():
                    code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                    code.writeline("block_stop = False")
                    code.writeline("while not block_stop:")
                    with code.indent():
                        code.writeline(
                            "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                        )
                        code.writeline(
                            "res = tl.where(stop, cur_inp, cur_inp + cur_src)"
                        )
                        code.writeline(
                            "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res)"
                        )
                        code.writeline("stop |= cur_inp == cas_res")
                        code.writeline(
                            "block_stop = tl.sum(stop.to(tl.int32)) == BLOCK"
                        )

            # Product reduction using CAS loop
            code.writeline("elif IS_PROD:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, cur_inp * cur_src)")
                    code.writeline(
                        "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res)"
                    )
                    code.writeline("stop |= cur_inp == cas_res")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            # Max reduction using CAS loop (atomic_max doesn't support float)
            code.writeline("elif IS_AMAX:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where(cur_src > cur_inp, cur_src, cur_inp)"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res)"
                    )
                    code.writeline("stop |= cur_inp == cas_res")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            # Min reduction using CAS loop (atomic_min doesn't support float)
            code.writeline("elif IS_AMIN:")
            with code.indent():
                code.writeline("stop = tl.where(mask, 0, 1).to(tl.int1)")
                code.writeline("block_stop = False")
                code.writeline("while not block_stop:")
                with code.indent():
                    code.writeline(
                        "cur_inp = tl.load(out + inp_offsets, mask=mask, other=0)"
                    )
                    code.writeline(
                        "new_val = tl.where(cur_src < cur_inp, cur_src, cur_inp)"
                    )
                    code.writeline("res = tl.where(stop, cur_inp, new_val)")
                    code.writeline(
                        "cas_res = tl.atomic_cas(out + inp_offsets, cur_inp, res)"
                    )
                    code.writeline("stop |= cur_inp == cas_res")
                    code.writeline("block_stop = tl.sum(stop.to(tl.int32)) == BLOCK")

            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def generate_count_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    """Generate kernel to count elements for mean reduction."""
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("{")
        with code.indent():
            code.writeline('"BLOCK": heur_block,')
            code.writeline('"LOOP": loop_count,')
        code.writeline("}")
    code.writeline(")")
    inp_stride_vars = ",".join(f"'inp_stride_{i}'" for i in range(rank))
    index_stride_vars = ",".join(f"'index_stride_{i}'" for i in range(rank))
    shape_vars = ",".join(f"'shape_{i}'" for i in range(rank))
    code.writeline(
        f"@triton.jit(do_not_specialize=['N','stride_dim','inp_size_dim',"
        f"{inp_stride_vars},{index_stride_vars},{shape_vars}])"
    )

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("count,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(f"shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape")
            code.writeline("inp_size_dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("BLOCK: tl.constexpr,")
            code.writeline("LOOP: tl.constexpr,")
            code.writeline("INT32_OFFSET: tl.constexpr")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("if not INT32_OFFSET:")
        with code.indent():
            code.writeline("pid = pid.to(tl.int64)")
        code.writeline("offsets = pid * LOOP * BLOCK + tl.arange(0, BLOCK)")

        code.writeline("for loop_iter in tl.static_range(LOOP):")
        with code.indent():
            code.writeline("mask = offsets < N")
            code.writeline("cur_idx = offsets")
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int32)")
            code.writeline("else:")
            with code.indent():
                code.writeline("inp_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
                code.writeline("idx_offsets = tl.zeros((BLOCK, ), dtype=tl.int64)")
            for i in range(rank)[::-1]:
                code.writeline("if INT32_OFFSET:")
                with code.indent():
                    code.writeline(f"shape_{i} = shape_{i}.to(tl.int32)")
                    code.writeline(f"inp_stride_{i} = inp_stride_{i}.to(tl.int32)")
                    code.writeline(f"index_stride_{i} = index_stride_{i}.to(tl.int32)")
                code.writeline(f"mod = cur_idx % shape_{i}")
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
                code.writeline(f"idx_offsets += mod * index_stride_{i}")
                if i != 0:
                    code.writeline(f"cur_idx = cur_idx // shape_{i}")

            code.writeline(
                "cur_index = tl.load(index + idx_offsets, mask=mask, other=0)"
            )
            code.writeline("if INT32_OFFSET:")
            with code.indent():
                code.writeline("cur_index = cur_index.to(tl.int32)")
                code.writeline("stride_dim = stride_dim.to(tl.int32)")

            code.writeline("dim_offsets = cur_index * stride_dim")
            code.writeline("inp_offsets += dim_offsets")
            code.newline()

            # Add 1 for each element
            code.writeline("one = tl.full((BLOCK,), 1, dtype=tl.int32)")
            code.writeline("tl.atomic_add(count + inp_offsets, one, mask=mask)")
            code.writeline("offsets += BLOCK")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim_size")
    parameters.append("dim_stride")
    parameters.append("N")
    parameters.append("reduce: tl.constexpr=None")
    parameters.append("include_self: tl.constexpr=True")
    parameters.append("int32_offset: tl.constexpr=None")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    count_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("index_strides = index.stride()")
        code.writeline("src_strides = src_strided.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("inp_size_dim = dim_size")
        code.writeline("stride_dim = dim_stride")

        code.writeline('IS_SUM = reduce == "sum"')
        code.writeline('IS_PROD = reduce == "prod"')
        code.writeline('IS_AMAX = reduce == "amax"')
        code.writeline('IS_AMIN = reduce == "amin"')
        code.writeline('IS_MEAN = reduce == "mean"')
        code.writeline("IS_FLOAT32 = out.dtype == torch.float32")
        code.writeline("int32_offset = int32_offset or True")

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK"] * meta["LOOP"]), ')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("src_strided, index, inp, out, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("inp_size_dim,")
                code.writeline("stride_dim,")
                code.writeline("N,")
                # reduce options
                code.writeline("IS_SUM,")
                code.writeline("IS_PROD,")
                code.writeline("IS_AMAX,")
                code.writeline("IS_AMIN,")
                code.writeline("IS_MEAN,")
                code.writeline("IS_FLOAT32,")
                code.writeline("INT32_OFFSET=int32_offset,")
        code.writeline(")")

        # Handle mean: need to divide by count
        code.writeline("if IS_MEAN:")
        with code.indent():
            code.writeline("count = torch.zeros_like(out, dtype=torch.int32)")
            # if include_self, initialize count to 1
            code.writeline("if include_self:")
            with code.indent():
                code.writeline("count.fill_(1)")
            count_launch: str = f"{count_kernel_name}[grid]("
            code.writeline(count_launch)
            with code.indent():
                code.writeline("index, count, ")
                if rank > 0:
                    s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_strides[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                    code.writeline(f"{s},")

                    code.writeline("inp_size_dim,")
                    code.writeline("stride_dim,")
                    code.writeline("N,")
                    code.writeline("INT32_OFFSET=int32_offset,")
            code.writeline(")")
            code.writeline("count = count.clamp(min=1)")
            code.writeline("out.div_(count)")

        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    count_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_reduce_kernel(rank, kernel_name, code)
    code = generate_count_kernel(rank, count_kernel_name, code)
    code = generate_destination_passing_wrapper(
        rank, wrapper_name, kernel_name, count_kernel_name, code
    )
    return code


class ScatterReduceFunction:
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
                "_scatter_reduce_wrapper",
                "_scatter_reduce_jit_function",
                "_scatter_reduce_count_jit_function",
                code,
            )

            file_name = f"scatter_reduce_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_scatter_reduce_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_reduce_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_reduce_func = ScatterReduceFunction()


def _get_init_value(reduce: str, dtype: torch.dtype, include_self: bool):
    """Get the initial value for reduction when include_self=False."""
    if include_self:
        return None  # No initialization needed, use original values

    if reduce == "sum":
        return 0
    elif reduce == "prod":
        return 1
    elif reduce == "amax":
        if dtype.is_floating_point:
            return float("-inf")
        else:
            return torch.iinfo(dtype).min
    elif reduce == "amin":
        if dtype.is_floating_point:
            return float("inf")
        else:
            return torch.iinfo(dtype).max
    elif reduce == "mean":
        return 0
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")


# ---------------------------------------------------------------------------
# GCU400 fast path: treat index as a [num_rows, num_cols] row-major 2D view
# ---------------------------------------------------------------------------
# The generic codegen kernel performs a multidimensional coordinate decomposition (integer div + mod) for every element.
# GCU400 has no integer-division hardware, so per-element div/mod becomes the main cost.
# With the 2D view:
#   * the row number comes from the program id, and the decomposition of the row number -> outer-dimension coordinates
# is computed only once per program;
#   * the intra-row column offset is a constant stride and accesses to src/index within a row are fully contiguous,
# which helps memory coalescing.
# The landing position of each reduce along dim is uniformly written as (dim_stride is the true stride of inp along
# dim):
#   out_off = inp_row_off + col * inp_stride_last + (index_val - col) * dim_stride
# When dim is the last dimension, dim_stride == inp_stride_last and the col term cancels out automatically.
_MAX_INT32_OFFSET = 2**31 - 1

# Tuning knobs for the tiling parameters (values are empirically optimal on GCU400)
_SR_BLOCK_M = 1
_SR_BLOCK_N = 512
_SR_NUM_WARPS = 4
_SR_NUM_STAGES = 1


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_cols",
        "shape_last_outer",
        "inp_stride_outer0",
        "inp_stride_outer1",
        "src_stride_outer0",
        "src_stride_outer1",
        "inp_stride_last",
        "dim_stride",
    ]
)
def _scatter_reduce_2d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    num_rows,
    num_cols,
    shape_last_outer,
    inp_stride_outer0,
    inp_stride_outer1,
    src_stride_outer0,
    src_stride_outer1,
    inp_stride_last,
    dim_stride,
    N_OUTER: tl.constexpr,
    IS_SUM: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_PROD: tl.constexpr,
    IS_AMAX: tl.constexpr,
    IS_AMIN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_ids = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    if INT32_OFFSET:
        row_ids = row_ids.to(tl.int32)
        col_ids = col_ids.to(tl.int32)
    else:
        row_ids = row_ids.to(tl.int64)
        col_ids = col_ids.to(tl.int64)

    row_mask = row_ids < num_rows
    col_mask = col_ids < num_cols

    # row number -> outer-dimension coordinates (row-major), done only once per program
    if N_OUTER == 2:
        coord_last_outer = row_ids % shape_last_outer
        coord_first_outer = row_ids // shape_last_outer
        inp_row_off = (
            coord_first_outer * inp_stride_outer0 + coord_last_outer * inp_stride_outer1
        )
        src_row_off = (
            coord_first_outer * src_stride_outer0 + coord_last_outer * src_stride_outer1
        )
    else:
        inp_row_off = row_ids * inp_stride_outer0
        src_row_off = row_ids * src_stride_outer0

    mask = row_mask[:, None] & col_mask[None, :]
    src_offsets = src_row_off[:, None] + col_ids[None, :]
    idx_offsets = row_ids[:, None] * num_cols + col_ids[None, :]

    num_valid = BLOCK_M * BLOCK_N
    if INT32_OFFSET:
        stop = tl.where(mask, 0, 1).to(tl.int1)

    cur_src = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
        stride_dim = dim_stride.to(tl.int32)
        stride_last = inp_stride_last.to(tl.int32)
    else:
        stride_dim = dim_stride
        stride_last = inp_stride_last

    out_offsets = (
        inp_row_off[:, None]
        + col_ids[None, :] * stride_last
        + (cur_index - col_ids[None, :]) * stride_dim
    )
    # Addresses of out-of-bounds lanes must be zeroed: GCU's atomic still resolves addresses for masked lanes,
    # so a directly computed negative offset would trigger an illegal device-side memory access
    out_offsets = tl.where(mask, out_offsets, 0)

    if IS_SUM | IS_MEAN:
        tl.atomic_add(out_ptr + out_offsets, cur_src, mask=mask)
    elif IS_PROD:
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out_ptr + out_offsets, mask=mask, other=0)
            res = tl.where(stop, cur_out, cur_out * cur_src)
            cas_res = tl.atomic_cas(out_ptr + out_offsets, cur_out, res)
            stop |= cur_out == cas_res
            block_stop = tl.sum(stop.to(tl.int32)) == num_valid
    elif IS_AMAX:
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out_ptr + out_offsets, mask=mask, other=0)
            new_val = tl.where(cur_src > cur_out, cur_src, cur_out)
            res = tl.where(stop, cur_out, new_val)
            cas_res = tl.atomic_cas(out_ptr + out_offsets, cur_out, res)
            stop |= cur_out == cas_res
            block_stop = tl.sum(stop.to(tl.int32)) == num_valid
    elif IS_AMIN:
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out_ptr + out_offsets, mask=mask, other=0)
            new_val = tl.where(cur_src < cur_out, cur_src, cur_out)
            res = tl.where(stop, cur_out, new_val)
            cas_res = tl.atomic_cas(out_ptr + out_offsets, cur_out, res)
            stop |= cur_out == cas_res
            block_stop = tl.sum(stop.to(tl.int32)) == num_valid


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_cols",
        "shape_last_outer",
        "inp_stride_outer0",
        "inp_stride_outer1",
        "inp_stride_last",
        "dim_stride",
    ]
)
def _scatter_reduce_2d_count_kernel(
    index_ptr,
    count_ptr,
    num_rows,
    num_cols,
    shape_last_outer,
    inp_stride_outer0,
    inp_stride_outer1,
    inp_stride_last,
    dim_stride,
    N_OUTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_ids = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
    col_ids = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    if INT32_OFFSET:
        row_ids = row_ids.to(tl.int32)
        col_ids = col_ids.to(tl.int32)
        stride_dim = dim_stride.to(tl.int32)
        stride_last = inp_stride_last.to(tl.int32)
    else:
        row_ids = row_ids.to(tl.int64)
        col_ids = col_ids.to(tl.int64)
        stride_dim = dim_stride
        stride_last = inp_stride_last

    mask = (row_ids < num_rows)[:, None] & (col_ids < num_cols)[None, :]

    if N_OUTER == 2:
        coord_last_outer = row_ids % shape_last_outer
        coord_first_outer = row_ids // shape_last_outer
        inp_row_off = (
            coord_first_outer * inp_stride_outer0 + coord_last_outer * inp_stride_outer1
        )
    else:
        inp_row_off = row_ids * inp_stride_outer0

    idx_offsets = row_ids[:, None] * num_cols + col_ids[None, :]
    cur_index = tl.load(index_ptr + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)

    out_offsets = (
        inp_row_off[:, None]
        + col_ids[None, :] * stride_last
        + (cur_index - col_ids[None, :]) * stride_dim
    )
    out_offsets = tl.where(mask, out_offsets, 0)
    tl.atomic_add(count_ptr + out_offsets, tl.where(mask, 1, 0), mask=mask)


def _scatter_reduce_2d_plan(index, src_strided, inp_restrided, dim_stride):
    """Returns the tiling metadata when the 2D-view fast path applies, otherwise None."""
    rank = index.dim()
    if rank == 0 or rank > 3:
        return None
    # Row-major flattening requires index to be fully contiguous; src must also have unit stride within a row
    if not index.is_contiguous() or src_strided.stride(-1) != 1:
        return None
    num_elements = index.numel()
    if num_elements == 0 or index.shape[-1] == 0:
        return None
    if max(num_elements, inp_restrided.numel()) > _MAX_INT32_OFFSET:
        return None

    shape = index.shape
    num_cols = shape[-1]
    num_rows = num_elements // num_cols
    inp_strides = inp_restrided.stride()
    src_strides = src_strided.stride()

    if rank == 1:
        n_outer, shape_last_outer = 0, 1
        inp_outer0 = inp_outer1 = src_outer0 = src_outer1 = 0
    elif rank == 2:
        n_outer, shape_last_outer = 1, 1
        inp_outer0, inp_outer1 = inp_strides[0], 0
        src_outer0, src_outer1 = src_strides[0], 0
    else:
        n_outer, shape_last_outer = 2, shape[1]
        inp_outer0, inp_outer1 = inp_strides[0], inp_strides[1]
        src_outer0, src_outer1 = src_strides[0], src_strides[1]

    return (
        n_outer,
        num_rows,
        num_cols,
        shape_last_outer,
        inp_outer0,
        inp_outer1,
        src_outer0,
        src_outer1,
        inp_strides[-1],
        dim_stride,
    )


def _scatter_reduce_2d_launch(
    out, index, src_strided, inp_restrided, dim_stride, reduce, include_self
):
    plan = _scatter_reduce_2d_plan(index, src_strided, inp_restrided, dim_stride)
    if plan is None:
        return False

    (
        n_outer,
        num_rows,
        num_cols,
        shape_last_outer,
        inp_outer0,
        inp_outer1,
        src_outer0,
        src_outer1,
        inp_stride_last,
        dim_stride,
    ) = plan

    flags = dict(
        N_OUTER=n_outer,
        IS_SUM=reduce == "sum",
        IS_MEAN=reduce == "mean",
        IS_PROD=reduce == "prod",
        IS_AMAX=reduce == "amax",
        IS_AMIN=reduce == "amin",
        BLOCK_M=_SR_BLOCK_M,
        BLOCK_N=_SR_BLOCK_N,
        INT32_OFFSET=True,
    )

    grid = (
        triton.cdiv(num_rows, _SR_BLOCK_M),
        triton.cdiv(num_cols, _SR_BLOCK_N),
    )
    launch_kwargs = dict(num_warps=_SR_NUM_WARPS, num_stages=_SR_NUM_STAGES)

    if reduce == "mean":
        # mean needs the contribution count for each landing position: accumulate the counts first, then divide once at
        # the end
        count = torch.zeros_like(out, dtype=torch.int32)
        if include_self:
            count.fill_(1)
        _scatter_reduce_2d_count_kernel[grid](
            index,
            count,
            num_rows,
            num_cols,
            shape_last_outer,
            inp_outer0,
            inp_outer1,
            inp_stride_last,
            dim_stride,
            **flags,
            **launch_kwargs,
        )
        count = count.clamp(min=1)

    _scatter_reduce_2d_kernel[grid](
        src_strided,
        index,
        out,
        num_rows,
        num_cols,
        shape_last_outer,
        inp_outer0,
        inp_outer1,
        src_outer0,
        src_outer1,
        inp_stride_last,
        dim_stride,
        **flags,
        **launch_kwargs,
    )

    if reduce == "mean":
        out.div_(count)
    return True


def _finish_inplace(original, work, needs_writeback):
    """Return the caller's tensor, honouring the in-place contract.

    The reduce kernels accumulate in fp32, so fp16/bf16 inputs are processed in a
    temporary workspace. Without the copy-back the caller's tensor would keep its
    old contents while the result would only exist in the returned tensor, which
    breaks ``scatter_reduce_`` semantics. The copy back replaces the dtype
    conversion that used to build the return value, so it costs nothing extra.
    """
    if needs_writeback:
        original.copy_(work)
        return original
    return work


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS_ENFLAME SCATTER_REDUCE_")
    original = inp
    needs_writeback = inp.dtype == torch.float16 or inp.dtype == torch.bfloat16
    if needs_writeback:
        inp = inp.to(torch.float32)
    out = inp

    # Validate reduce operation
    assert reduce in (
        "sum",
        "prod",
        "mean",
        "amax",
        "amin",
    ), f"Unsupported reduce operation: {reduce}"

    # Check for internal overlapping
    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."

    # Handle include_self=False: initialize with identity values
    if not include_self:
        init_value = _get_init_value(reduce, inp.dtype, include_self)
        if init_value is not None:
            out.fill_(init_value)

    src_restrided = src.as_strided(index.shape, src.stride())
    inp_restrided = restride_dim(inp, dim, index.shape)
    dim_size = inp.size(dim)
    dim_stride = inp.stride(dim)
    N = index.numel()

    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (inp, index, src)))

    if _scatter_reduce_2d_launch(
        out, index, src_restrided, inp_restrided, dim_stride, reduce, include_self
    ):
        return _finish_inplace(original, inp, needs_writeback)

    _scatter_reduce_func(
        src_restrided,
        index,
        inp_restrided,
        out,
        dim_size,
        dim_stride,
        N,
        reduce,
        include_self,
        int32_offset=use_int32_offset,
    )

    return _finish_inplace(original, inp, needs_writeback)
