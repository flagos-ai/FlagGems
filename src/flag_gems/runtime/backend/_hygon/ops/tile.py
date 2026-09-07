import importlib
import logging
import os
from typing import Callable, List, Mapping

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


# --------------------------- tile wrapper genration -----------------------------------
def parameter_for_wrapper() -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("dims")
    return ", ".join(parameters)


def parameter_for_wrapper_out() -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("out0")

    return ", ".join(parameters)


def parameter_ref_for_wrapper() -> str:
    """Generate parameter reference for wrapper function.
    Example: in0, val0, out0, out0_offset
    """
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
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")
    code.newline()
    code.newline()
    return code


def generate_functional_tile_wrapper(
    wrapper_name: str,
    destination_passing_func_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
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
                "assert(dims_shape[i] >= 0), 'the number of repetitions per dimension out of range (expected to >= 0) \
                but got {}'.format(dims_shape[i])"
            )
            code.writeline("out_dim = in0_shape[i] * dims_shape[i]")
            code.writeline("if out_dim == 0:")
            with code.indent():
                code.writeline("is_empty = True")
            code.writeline("out_shape.append(out_dim)")
        code.newline()
        code.writeline(
            "out0 = torch.empty(out_shape, device=in0.device, dtype=in0.dtype)"
        )

        code.writeline("in0 = in0.reshape(in0_shape)")
        code.writeline("if not is_empty:")
        with code.indent():
            # call destination_passing_func
            output_names: str = output_ref_for_wrapper()
            call_str = (
                f"{output_names} = {destination_passing_func_name}"
                f"({parameter_ref_for_wrapper()})"
            )
            code.writeline(call_str)

        code.writeline("return out0")
        code.newline()
        code.newline()

    return code


def generate_destination_passing_tile_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    contiguous_kernel_name: str,
    contiguous_block_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper_out()

    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        if rank == 0:
            code.writeline("out0.copy_(in0)")
            code.writeline("return out0")
            code.newline()
            code.newline()
            return code

        code.writeline("shape = out0.shape")
        code.writeline("num_tasks = volume(shape)")
        code.writeline("if num_tasks == 0:")
        with code.indent():
            code.writeline("return out0")
        code.newline()
        # code.writeline("tile_size = min(512, triton.next_power_of_2(num_tasks))")
        code.writeline("tile_size = min(1024, triton.next_power_of_2(num_tasks))")
        # code.writeline("num_warps = 4")
        code.writeline("num_warps = 1 if tile_size <= 64 else 8")
        code.writeline("num_ctas = min(65535, triton.cdiv(num_tasks, tile_size))")
        code.writeline("tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)")
        code.writeline("grid = (num_ctas, 1, 1)")
        code.newline()

        code.writeline("# strides and shapes for the generic fallback")
        code.writeline("in0_strides = in0.stride()")
        code.writeline("in0_shape = in0.shape")
        code.writeline("out0_strides = out0.stride()")
        code.writeline("in0_numel = in0.numel()")
        code.newline()

        code.writeline("# kernel launch")
        code.writeline("with torch_device_fn.device(in0.device.index):")
        with code.indent():
            code.writeline("if in0.is_contiguous() and out0.is_contiguous():")
            with code.indent():
                if rank == 1:
                    code.writeline("use_block_kernel = True")
                else:
                    trailing_pairs = " and ".join(
                        f"shape[{i}] == in0_shape[{i}]" for i in range(1, rank)
                    )
                    code.writeline(f"use_block_kernel = {trailing_pairs}")
                code.writeline("if use_block_kernel:")
                with code.indent():
                    code.writeline(f"{contiguous_block_kernel_name}[grid](")
                    with code.indent():
                        code.writeline("in0, out0,")
                        code.writeline("in0_numel,")
                        code.writeline("num_tasks,")
                        code.writeline("tiles_per_cta=tiles_per_cta,")
                        code.writeline("tile_size=tile_size,")
                        code.writeline("one_tile_per_cta=tiles_per_cta == 1,")
                        code.writeline("num_warps=num_warps,")
                    code.writeline(")")
                code.writeline("else:")
                with code.indent():
                    code.writeline(f"{contiguous_kernel_name}[grid](")
                    with code.indent():
                        code.writeline("in0, out0,")
                        shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                        code.writeline(f"{shape_args}, # output shape")
                        in_shape_args: str = ", ".join(
                            f"in0_shape[{i}]" for i in range(rank)
                        )
                        code.writeline(f"{in_shape_args}, # input shape")
                        code.writeline("num_tasks,")
                        code.writeline("tiles_per_cta=tiles_per_cta,")
                        code.writeline("tile_size=tile_size,")
                        code.writeline("one_tile_per_cta=tiles_per_cta == 1,")
                        code.writeline("num_warps=num_warps,")
                    code.writeline(")")

            code.writeline("else:")
            with code.indent():
                code.writeline(f"{kernel_name}[grid](")
                with code.indent():
                    code.writeline("in0, out0,")

                    s = ", ".join(f"in0_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for in0")

                    s = ", ".join(f"out0_strides[{j}]" for j in range(rank))
                    code.writeline(f"{s}, # stride for out0")

                    shape_args: str = ", ".join(f"shape[{i}]" for i in range(rank))
                    code.writeline(f"{shape_args}, # task indexing space")
                    in_shape_args: str = ", ".join(
                        f"in0_shape[{i}]" for i in range(rank)
                    )
                    code.writeline(
                        f"{in_shape_args}, # task indexing space used when input and output tensor has different shape"
                    )
                    code.writeline("num_tasks, # num tasks")
                    code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                    code.writeline("tile_size=tile_size,")
                    code.writeline("one_tile_per_cta=tiles_per_cta == 1,")
                    code.writeline("num_warps=num_warps,")
                code.writeline(")")

        code.writeline("return out0")
        code.newline()
        code.newline()
    return code


def generate_contiguous_block_tile_kernel(
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
        code.writeline("in0_numel: tl.int64,")
        code.writeline("num_tasks: tl.int64,")
        code.writeline("tiles_per_cta: tl.int64,")
        code.writeline("tile_size: tl.constexpr,")
        code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = ext.program_id(0)")
        code.writeline("num_ctas = ext.num_programs(0)")
        code.writeline("init_tid = pid * tile_size + tl.arange(0, tile_size)")
        code.newline()
        code.writeline("if one_tile_per_cta:")
        with code.indent():
            code.writeline("tid = init_tid")
            code.writeline("mask = tid < num_tasks")
            code.writeline("in_offset = tid % in0_numel")
            code.writeline("out0 = tl.load(in0_ptr + in_offset, mask=mask)")
            code.writeline("tl.store(out0_ptr + tid, out0, mask=mask)")
        code.writeline("else:")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                code.writeline("tid = init_tid + j * tile_size * num_ctas")
                code.writeline("mask = tid < num_tasks")
                code.writeline("in_offset = tid % in0_numel")
                code.writeline("out0 = tl.load(in0_ptr + in_offset, mask=mask)")
                code.writeline("tl.store(out0_ptr + tid, out0, mask=mask)")
        code.newline()
    return code


def _contiguous_stride_expr(rank: int, dim: int) -> str:
    if dim == rank - 1:
        return "1"
    return " * ".join(f"in_s{i}" for i in range(dim + 1, rank))


def _write_contiguous_tile_body(code: IndentedBuffer, rank: int) -> None:
    code.writeline("linear_tid = tid")
    code.writeline("idx = tid")
    code.writeline("in_offset = linear_tid * 0")
    code.newline()
    code.writeline("# reconstruct only the contiguous input offset")
    for i in reversed(range(rank)):
        if i > 0:
            code.writeline(f"i{i} = idx % s{i}")
            code.writeline(f"idx //= s{i}")
        else:
            code.writeline("i0 = idx")
    code.newline()

    for i in range(rank):
        stride_expr = _contiguous_stride_expr(rank, i)
        code.writeline(f"if in_s{i} != 1:")
        with code.indent():
            code.writeline(f"if s{i} == in_s{i}:")
            with code.indent():
                code.writeline(f"in_i{i} = i{i}")
            code.writeline("else:")
            with code.indent():
                code.writeline(f"in_i{i} = i{i} % in_s{i}")
            if stride_expr == "1":
                code.writeline(f"in_offset += in_i{i}")
            else:
                code.writeline(f"in_offset += in_i{i} * ({stride_expr})")
    code.newline()
    code.writeline("out0 = tl.load(in0_ptr + in_offset, mask=mask)")
    code.writeline("tl.store(out0_ptr + linear_tid, out0, mask=mask)")


def generate_contiguous_tile_kernel(
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
        shape_args = ", ".join(f"s{i}: tl.constexpr" for i in range(rank))
        code.writeline(f"{shape_args}, # output shape")
        in_shape_args = ", ".join(f"in_s{i}: tl.constexpr" for i in range(rank))
        code.writeline(f"{in_shape_args}, # input shape")
        code.writeline("num_tasks: tl.int64,")
        code.writeline("tiles_per_cta: tl.int64,")
        code.writeline("tile_size: tl.constexpr,")
        code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("pid = ext.program_id(0)")
        code.writeline("num_ctas = ext.num_programs(0)")
        code.writeline("init_tid = pid * tile_size + tl.arange(0, tile_size)")
        code.newline()
        code.writeline("if one_tile_per_cta:")
        with code.indent():
            code.writeline("tid = init_tid")
            code.writeline("mask = tid < num_tasks")
            _write_contiguous_tile_body(code, rank)
        code.writeline("else:")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                code.writeline("tid = init_tid + j * tile_size * num_ctas")
                code.writeline("mask = tid < num_tasks")
                _write_contiguous_tile_body(code, rank)
                code.newline()
    return code


def generate_tile_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        # signature: inputs ptrs & non tensor inputs
        code.writeline("in0_ptr: tl.tensor, # of tl.pointer_type")

        # signature: output ptrs
        code.writeline("out0_ptr: tl.tensor, # of tl.pointer_type")

        stride_args = ", ".join(f"in0_stride{j}: tl.int64" for j in range(rank))
        code.writeline(f"{stride_args}, # strides for in0")

        stride_args = ", ".join(f"out0_stride{j}: tl.int64" for j in range(rank))
        code.writeline(f"{stride_args}, # strides for out0")

        task_space_args = ", ".join(f"s{i}: tl.int64" for i in range(rank))
        code.writeline(f"{task_space_args}, # task_space")

        task_space_args2 = ", ".join(f"in_s{i}: tl.int64" for i in range(rank))
        code.writeline(
            f"{task_space_args2}, # task_space2 used when input and output tensor has different shape"
        )

        code.writeline("num_tasks: tl.int64,")
        code.writeline("tiles_per_cta: tl.int64,")
        code.writeline("tile_size: tl.constexpr,")
        code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        code.writeline("# task id & masking")
        code.writeline("pid = ext.program_id(0)")
        code.writeline("num_ctas = ext.num_programs(0)")
        code.writeline("init_tid = pid * tile_size + tl.arange(0, tile_size)")

        code.writeline("if one_tile_per_cta: # monolitic kernel style")
        with code.indent():
            code.writeline("tid = init_tid")
            code.writeline("mask = tid < num_tasks")
            code.newline()

            code.writeline("# multi index recontruction")
            for i in reversed(range(rank)):
                if i > 0:
                    code.writeline(f"i{i} = tid % s{i}")
                    code.writeline(f"tid //= s{i}")
                else:
                    code.writeline(f"i{i} = tid")
            code.newline()

            code.writeline("# loads")
            ptrs_expr: str = " + ".join(
                f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank)
            )
            ptrs_expr = f"in0_ptr + {ptrs_expr}"
            code.writeline(f"in0 = tl.load({ptrs_expr}, mask=mask)")
            code.newline()

            code.writeline("# compute")
            code.writeline("out0 = in0")
            code.newline()

            code.writeline("# stores")
            ptrs_expr = " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
            ptrs_expr = f"out0_ptr + {ptrs_expr}"
            code.writeline(f"tl.store({ptrs_expr}, out0, mask=mask)")

        code.writeline("else: # grid-stride-loop style kernel")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                code.writeline("tid = init_tid + j * tile_size * num_ctas")
                code.writeline("mask = tid < num_tasks")
                code.newline()

                code.writeline("# multi index recontruction")
                for i in reversed(range(rank)):
                    if i > 0:
                        code.writeline(f"i{i} = tid % s{i}")
                        code.writeline(f"tid //= s{i}")
                    else:
                        code.writeline(f"i{i} = tid")
                code.newline()

                code.writeline("# loads")
                ptrs_expr = " + ".join(
                    f"(i{j} % in_s{j}) * in0_stride{j}" for j in range(rank)
                )
                ptrs_expr = f"in0_ptr + {ptrs_expr}"
                code.writeline(f"in0 = tl.load({ptrs_expr}, mask=mask)")
                code.newline()

                code.writeline("# compute")
                code.writeline("out0 = in0")
                code.newline()

                code.writeline("# stores")
                ptrs_expr = " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
                ptrs_expr = f"out0_ptr + {ptrs_expr}"
                code.writeline(f"tl.store({ptrs_expr}, out0, mask=mask)")
                code.newline()
    return code


def generate_code(
    rank: int,
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    contiguous_kernel_name: str,
    contiguous_block_kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code = generate_imports(code)
    code = generate_functional_tile_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_tile_wrapper(
        rank,
        destination_passing_func_name,
        kernel_name,
        contiguous_kernel_name,
        contiguous_block_kernel_name,
        code,
    )
    if rank > 0:
        code = generate_contiguous_block_tile_kernel(contiguous_block_kernel_name, code)
        code = generate_contiguous_tile_kernel(rank, contiguous_kernel_name, code)
        code = generate_tile_kernel(rank, kernel_name, code)
    return code


class TileFunction:
    def __init__(self):
        self.pid = os.getpid()
        # instantiated & cached overloads
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, x, dims):
        # note: kwargs should not be used in JITFunction directly
        ndim = self.arg_key(x, dims)
        key = str(ndim)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            # generate file & import it
            code = IndentedBuffer()
            code = generate_code(
                ndim,
                "_wrapper",
                "_wrapper_out",
                "_tile_flaggems_jit_function",
                "_tile_contiguous_flaggems_jit_function",
                "_tile_contiguous_block_flaggems_jit_function",
                code,
            )

            file_name = f"tile_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(x, dims)

    def arg_key(self, x, dims):
        max_rank = max(x.ndim, len(dims))
        return max_rank


_tile_func = TileFunction()


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logger.debug("GEMS TILE")

    out = _tile_func(inp, dims)
    return out
