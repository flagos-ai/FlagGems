import importlib
import logging
import os
from typing import Callable, List, Mapping

import torch
import triton
from triton import language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["in_numel", "tile_factor"])
def tile_first_dim_kernel(
    in_ptr, out_ptr, in_numel, tile_factor,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = tl.cdiv(in_numel, BLOCK)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < in_numel
        val = tl.load(in_ptr + off, mask=mask)
        for k in tl.range(0, tile_factor):
            tl.store(out_ptr + k * in_numel + off, val, mask=mask)


def _tile_first_dim(inp, out, tile_factor):
    in_numel = inp.numel()
    is_fp32 = inp.dtype == torch.float32
    if in_numel <= 32768:
        BLOCK = max(1024, triton.next_power_of_2(in_numel))
    elif is_fp32:
        BLOCK = 32768
    else:
        BLOCK = 65536
    NUM_BLOCKS = triton.cdiv(in_numel, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        tile_first_dim_kernel[(grid_size,)](
            inp, out, in_numel, tile_factor,
            BLOCK=BLOCK, num_warps=1,
        )
    return out


def _can_use_flat_copy(inp, dims_shape, in_shape):
    if not inp.is_contiguous():
        return False
    first_dim_tiled = False
    for i, d in enumerate(dims_shape):
        if d > 1:
            if i == 0:
                first_dim_tiled = True
            else:
                return False
        elif d == 0:
            return False
    return first_dim_tiled or all(d == 1 for d in dims_shape)


# ============= Codegen-based general path =============

def parameter_for_wrapper() -> str:
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("dims")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("out0")
    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("out0")
    return ", ".join(parameters)


def output_ref_for_wrapper() -> str:
    return "out0"


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import math")
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("from triton import language as tl")
    code.newline()
    code.writeline("from flag_gems.runtime import torch_device_fn")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils.libentry import libentry")
    code.writeline("from flag_gems.utils.type_utils import type_promotion")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.writeline(f"NUM_SIPS = {NUM_SIPS}")
    code.newline()
    code.newline()
    return code


def generate_functional_tile_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("in0_rank = in0.dim()")
        code.writeline("dims_rank = len(dims)")
        code.writeline("in0_shape = list(in0.shape)")
        code.writeline("dims_shape = list(dims)")
        code.newline()
        code.writeline("if (dims_rank < in0_rank): ")
        with code.indent():
            code.writeline("diff = in0_rank - dims_rank")
            code.writeline("ones = [1 for _ in range(diff)]")
            code.writeline("dims_shape = ones + dims_shape")
        code.writeline("elif (dims_rank > in0_rank): ")
        with code.indent():
            code.writeline("diff = dims_rank - in0_rank")
            code.writeline("ones = [1 for _ in range(diff)]")
            code.writeline("in0_shape = ones + in0_shape")
        code.newline()
        code.writeline("is_empty = False")
        code.writeline("out_shape = []")
        code.writeline("for i in range(len(in0_shape)): ")
        with code.indent():
            code.writeline(
                "assert(dims_shape[i] >= 0), 'the number of repetitions per dimension out of range (expected to >= 0) "
                "but got {}'.format(dims_shape[i])"
            )
            code.writeline("if dims_shape[i] == 0: ")
            with code.indent():
                code.writeline("is_empty = True")
            code.writeline("out_shape.append(in0_shape[i] * dims_shape[i])")
        code.newline()
        code.writeline(
            "out0 = torch.empty(out_shape, device=in0.device, dtype=in0.dtype)"
        )

        code.writeline("in0 = in0.reshape(in0_shape)")
        code.writeline("if not is_empty: ")
        with code.indent():
            output_names: str = output_ref_for_wrapper()
            call_str = (
                f"{output_names} = {destination_passing_func_name}"
                f"({parameter_ref_for_wrapper()})"
            )
            code.writeline(call_str)

        return_str = "return out0"
        code.writeline(return_str)
        code.newline()
        code.newline()

    return code


def generate_destination_passing_tile_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper_out()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        if rank > 0:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = volume(shape)")

        if rank > 0:
            code.writeline(
                "tile_size = max(1024, min(65536, triton.next_power_of_2(num_tasks)))"
            )
            code.writeline("num_warps = 1")
            code.writeline(
                "num_ctas = min(NUM_SIPS * 2, triton.cdiv(num_tasks, tile_size))"
            )
            code.writeline(
                "tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)"
            )
        else:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        code.writeline("grid = (num_ctas, 1, 1)")
        code.newline()

        if rank > 0:
            code.writeline("in0_strides = in0.stride()")
            code.writeline("in0_shape = in0.shape")
            code.writeline("out0_strides = out0.stride()")
        code.newline()

        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            kernel_launch: str = f"{kernel_name}[grid]("
            code.writeline(kernel_launch)

            with code.indent():
                code.writeline("in0, out0, ")

            if rank > 0:
                s = ", ".join(f"in0_strides[{j}]" for j in range(rank))
                code.writeline(f"{s}, # stride for in0")

                s = ", ".join(f"out0_strides[{j}]" for j in range(rank))
                code.writeline(f"{s}, # stride for out0")

                shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                code.writeline(f"{shape_args}, # task indexing space")
                in_shape_args: str = ", ".join(f"in0_shape[{i}]" for i in range(rank))
                code.writeline(
                    f"{in_shape_args}, # input shape for modular indexing"
                )
                code.writeline("num_tasks, # num tasks")
                code.writeline("tiles_per_cta=tiles_per_cta,")
                code.writeline("tile_size=tile_size,")
                code.writeline("one_tile_per_cta=tiles_per_cta==1,")
            code.writeline("num_warps=num_warps,")
        code.writeline(")")

        code.writeline("return out0")
        code.newline()
        code.newline()
    return code


def generate_tile_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("in0_ptr: tl.tensor,")
        code.writeline("out0_ptr: tl.tensor,")

        if rank > 0:
            stride_args = ", ".join(f"in0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args},")

            stride_args = ", ".join(f"out0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args},")

            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            code.writeline(f"{task_space_args},")

            task_space_args2 = ", ".join(f"in_s{i}: int" for i in range(rank))
            code.writeline(f"{task_space_args2},")

            code.writeline("num_tasks: int,")

        if rank > 0:
            code.writeline("tiles_per_cta,")
            code.writeline("tile_size: tl.constexpr,")
            code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tle.program_id(0)")
        code.writeline("num_ctas = tle.num_programs(0)")
        code.writeline("init_tid = pid * tile_size + tl.arange(0, tile_size)")

        code.writeline("if one_tile_per_cta:")
        with code.indent():
            code.writeline("tid = init_tid")
            code.writeline("mask = tid < num_tasks")
            code.newline()

            for i in reversed(range(rank)):
                if i > 0:
                    code.writeline(f"i{i} = tid % s{i}")
                    code.writeline(f"tid //= s{i}")
                else:
                    code.writeline(f"i{i} = tid")
            code.newline()

            ptrs_expr: str = " + ".join(
                f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank)
            )
            code.writeline(f"in0 = tl.load(in0_ptr + {ptrs_expr}, mask=mask)")
            code.newline()

            ptrs_expr: str = " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
            code.writeline(f"tl.store(out0_ptr + {ptrs_expr}, in0, mask=mask)")

        code.writeline("else:")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                code.writeline("tid = init_tid + j * tile_size * num_ctas")
                code.writeline("mask = tid < num_tasks")
                code.newline()

                for i in reversed(range(rank)):
                    if i > 0:
                        code.writeline(f"i{i} = tid % s{i}")
                        code.writeline(f"tid //= s{i}")
                    else:
                        code.writeline(f"i{i} = tid")
                code.newline()

                ptrs_expr: str = " + ".join(
                    f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank)
                )
                code.writeline(f"in0 = tl.load(in0_ptr + {ptrs_expr}, mask=mask)")
                code.newline()

                ptrs_expr: str = " + ".join(
                    f"i{j} * out0_stride{j}" for j in range(rank)
                )
                code.writeline(f"tl.store(out0_ptr + {ptrs_expr}, in0, mask=mask)")
                code.newline()
    return code


def generate_code(
    rank: int,
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code = generate_imports(code)
    code = generate_functional_tile_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_tile_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_tile_kernel(rank, kernel_name, code)
    return code


class TileFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, x, dims):
        ndim = self.arg_key(x, dims)

        in_shape = list(x.shape)
        dims_shape = list(dims)
        x_ndim = x.ndim
        d_ndim = len(dims_shape)
        if d_ndim < x_ndim:
            dims_shape = [1] * (x_ndim - d_ndim) + dims_shape
        elif d_ndim > x_ndim:
            in_shape = [1] * (d_ndim - x_ndim) + in_shape

        if _can_use_flat_copy(x, dims_shape, in_shape):
            is_empty = any(d == 0 for d in dims_shape)
            out_shape = [s * d for s, d in zip(in_shape, dims_shape)]
            out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
            if not is_empty and out.numel() > 0:
                tile_factor = dims_shape[0] if dims_shape[0] > 1 else 1
                inp_flat = x.reshape(in_shape).contiguous().view(-1)
                _tile_first_dim(inp_flat, out.view(-1), tile_factor)
            return out

        key = f"gcu400_{ndim}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                ndim,
                "_wrapper",
                "_wrapper_out",
                "_tile_flaggems_jit_function",
                code,
            )

            file_name = f"tile_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(x, dims)

    def arg_key(self, x, dims):
        max_rank = max(x.ndim, len(dims))
        return max_rank


_tile_func = TileFunction()


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS TILE GCU400")
    out = _tile_func(inp, dims)
    return out
