import importlib
import logging
import os
from typing import Callable, List, Mapping, Tuple, Union

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def parameter_for_wrapper() -> str:
    """Generate parameter declaration for wrapper function."""
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("shifts")
    parameters.append("dims")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    """Generate parameter declaration for wrapper function."""
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("out0")
    parameters.append("normalized_shifts")
    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    """Generate parameter reference for wrapper function."""
    parameters: List[str] = []
    parameters.append("in0")
    parameters.append("out0")
    parameters.append("normalized_shifts")
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
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_functional_roll_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("in0_rank = in0.dim()")
        code.writeline("in0_shape = list(in0.shape)")
        code.newline()

        code.writeline("# Normalize dims and shifts to lists")
        code.writeline("if dims is None:")
        with code.indent():
            code.writeline("# Flatten, roll on single dimension, reshape")
            code.writeline("original_shape = in0_shape")
            code.writeline("in0 = in0.reshape(-1)")
            code.writeline("in0_rank = 1")
            code.writeline("in0_shape = [in0.numel()]")
            code.writeline("dims = [0]")
            code.writeline("if isinstance(shifts, (list, tuple)):")
            with code.indent():
                code.writeline("shifts = [sum(shifts)]")
            code.writeline("else:")
            with code.indent():
                code.writeline("shifts = [shifts]")
        code.writeline("else:")
        with code.indent():
            code.writeline("original_shape = None")
            code.writeline("if isinstance(dims, int):")
            with code.indent():
                code.writeline("dims = [dims]")
            code.writeline("if isinstance(shifts, int):")
            with code.indent():
                code.writeline("shifts = [shifts]")
        code.newline()

        code.writeline("# Normalize negative dimensions")
        code.writeline("dims = [(d if d >= 0 else d + in0_rank) for d in dims]")
        code.newline()

        code.writeline("# Validate")
        code.writeline("assert len(shifts) == len(dims), \\")
        code.writeline(
            "    f'shifts and dimensions must align, got {len(shifts)} shifts and {len(dims)} dims'"
        )
        code.writeline("for d in dims:")
        with code.indent():
            code.writeline("assert 0 <= d < in0_rank, \\")
            code.writeline(
                "    f'Dimension out of range (expected to be in range of [0, {in0_rank-1}], but got {d})'"
            )
        code.newline()

        code.writeline("# Normalize shifts to be within [0, size) for each dimension")
        code.writeline("normalized_shifts = [0] * in0_rank")
        code.writeline("for shift, dim in zip(shifts, dims):")
        with code.indent():
            code.writeline("size = in0_shape[dim]")
            code.writeline("if size > 0:")
            with code.indent():
                code.writeline("# Python modulo handles negative shifts correctly")
                code.writeline("normalized_shifts[dim] = shift % size")
        code.newline()

        code.writeline("# Check if any rolling is needed")
        code.writeline("if all(s == 0 for s in normalized_shifts):")
        with code.indent():
            code.writeline("# No rolling needed, just clone")
            code.writeline("out0 = in0.clone()")
            code.writeline("if original_shape is not None:")
            with code.indent():
                code.writeline("out0 = out0.reshape(original_shape)")
            code.writeline("return out0")
        code.newline()

        code.writeline("out0 = torch.empty_like(in0)")
        code.newline()

        output_names: str = output_ref_for_wrapper()
        call_str = (
            f"{output_names} = {destination_passing_func_name}"
            f"({parameter_ref_for_wrapper()})"
        )
        code.writeline(call_str)
        code.newline()

        code.writeline("if original_shape is not None:")
        with code.indent():
            code.writeline("out0 = out0.reshape(original_shape)")

        return_str = "return out0"
        code.writeline(return_str)
        code.newline()
        code.newline()

    return code


def generate_destination_passing_roll_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(f"def {wrapper_name}(in0, out0, normalized_shifts):")

    with code.indent():
        if rank > 0:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = volume(shape)")
            code.newline()

            code.writeline("# Handle empty tensors")
            code.writeline("if num_tasks == 0:")
            with code.indent():
                code.writeline("return out0")
            code.newline()

        if rank > 0:
            code.writeline("tile_size = min(512, triton.next_power_of_2(num_tasks))")
            code.writeline("num_warps = 4")
            code.writeline("num_ctas = min(65535, triton.cdiv(num_tasks, tile_size))")
            code.writeline(
                "tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)"
            )
        else:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        code.writeline("grid = (num_ctas, 1, 1)")
        code.newline()

        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            code.writeline("in0_strides = in0.stride()")
            code.writeline("in0_shape = in0.shape")
            code.writeline("out0_strides = out0.stride()")
        code.newline()

        code.writeline("# kernel launch")
        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            kernel_launch: str = f"{kernel_name}[grid]("
            code.writeline(kernel_launch)

            with code.indent():
                code.writeline("in0, out0,")

                if rank > 0:
                    s = ", ".join(f"in0_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for in0")

                    s = ", ".join(f"out0_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for out0")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                    code.writeline(f"{shape_args}, # task indexing space")

                    shifts_args: str = ", ".join(
                        f"normalized_shifts[{i}]" for i in range(rank)
                    )
                    code.writeline(f"{shifts_args}, # shifts for each dimension")

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


def generate_roll_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    code.writeline(f"def {kernel_name}(")
    with code.indent():
        code.writeline("in0_ptr: tl.tensor, # of tl.pointer_type")
        code.writeline("out0_ptr: tl.tensor, # of tl.pointer_type")

        if rank > 0:
            stride_args = ", ".join(f"in0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # strides for in0")

            stride_args = ", ".join(f"out0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # strides for out0")

            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            code.writeline(f"{task_space_args}, # task_space")

            shifts_args = ", ".join(f"shift{i}: int" for i in range(rank))
            code.writeline(f"{shifts_args}, # shifts")

            code.writeline("num_tasks: int,")

        if rank > 0:
            code.writeline("tiles_per_cta,")
            code.writeline("tile_size: tl.constexpr,")
            code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("# task id & masking")
        code.writeline("pid = tle.program_id(0)")
        code.writeline("num_ctas = tle.num_programs(0)")

        code.writeline("init_tid = pid * tile_size + tl.arange(0, tile_size)")

        code.writeline("if one_tile_per_cta: # monolithic kernel style")
        with code.indent():
            code.writeline("tid = init_tid")
            code.writeline("mask = tid < num_tasks")
            code.newline()

            code.writeline("# multi index reconstruction")
            for i in reversed(range(rank)):
                if i > 0:
                    code.writeline(f"i{i} = tid % s{i}")
                    code.writeline(f"tid //= s{i}")
                else:
                    code.writeline(f"i{i} = tid")
            code.newline()

            code.writeline("# compute source indices with roll")
            for i in range(rank):
                # Use Python-like modulo: (i - shift) % size
                # In Triton, this is safe because we already normalized shifts in Python
                code.writeline(f"src_i{i} = (i{i} + s{i} - shift{i}) % s{i}")
            code.newline()

            code.writeline("# loads")
            ptrs_expr: str = " + ".join(
                f"src_i{j} * in0_stride{j}" for j in range(rank)
            )
            ptrs_expr: str = f"in0_ptr + {ptrs_expr}"
            code.writeline(f"in0 = tl.load({ptrs_expr}, mask=mask)")
            code.newline()

            code.writeline("# compute")
            code.writeline("out0 = in0")
            code.newline()

            code.writeline("# stores")
            ptrs_expr: str = " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
            ptrs_expr: str = f"out0_ptr + {ptrs_expr}"
            code.writeline(f"tl.store({ptrs_expr}, out0, mask=mask)")

        code.writeline("else: # grid-stride-loop style kernel")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                code.writeline("tid = init_tid + j * tile_size * num_ctas")
                code.writeline("mask = tid < num_tasks")
                code.newline()

                code.writeline("# multi index reconstruction")
                for i in reversed(range(rank)):
                    if i > 0:
                        code.writeline(f"i{i} = tid % s{i}")
                        code.writeline(f"tid //= s{i}")
                    else:
                        code.writeline(f"i{i} = tid")
                code.newline()

                code.writeline("# compute source indices with roll")
                for i in range(rank):
                    code.writeline(f"src_i{i} = (i{i} + s{i} - shift{i}) % s{i}")
                code.newline()

                code.writeline("# loads")
                ptrs_expr: str = " + ".join(
                    f"src_i{j} * in0_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"in0_ptr + {ptrs_expr}"
                code.writeline(f"in0 = tl.load({ptrs_expr}, mask=mask)")
                code.newline()

                code.writeline("# compute")
                code.writeline("out0 = in0")
                code.newline()

                code.writeline("# stores")
                ptrs_expr: str = " + ".join(
                    f"i{j} * out0_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"out0_ptr + {ptrs_expr}"
                code.writeline(f"tl.store({ptrs_expr}, out0, mask=mask)")
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
    code = generate_functional_roll_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_roll_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_roll_kernel(rank, kernel_name, code)
    return code


class RollFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[int, Callable] = {}

    def __call__(self, x, shifts, dims):
        ndim = self.arg_key(x, shifts, dims)
        key = str(ndim)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                ndim,
                "_wrapper",
                "_wrapper_out",
                "_roll_flaggems_jit_function",
                code,
            )

            file_name = f"roll_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_roll_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(x, shifts, dims)

    def arg_key(self, x, shifts, dims):
        if dims is None:
            return 1
        if isinstance(dims, int):
            return x.ndim
        return x.ndim


_roll_func = RollFunction()


def roll(
    inp: torch.Tensor,
    shifts: Union[int, Tuple[int, ...]],
    dims: Union[None, int, Tuple[int, ...]] = None,
) -> torch.Tensor:
    logger.debug("GEMS ROLL")
    out = _roll_func(inp, shifts, dims)
    return out
