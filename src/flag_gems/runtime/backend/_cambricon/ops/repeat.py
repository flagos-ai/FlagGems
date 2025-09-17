import importlib
import logging
import os
from typing import Callable, List, Mapping

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


# --------------------------- repeat wrapper genration -----------------------------------
def parameter_for_wrapper() -> str:
    """Generate parameter declaration with type annotation for wrapper function.
    Example: in0: torch.Tensor, val0: float, out0: torch.Tensor
    """
    parameters: List[str] = []

    parameters.append("in0")
    parameters.append("sizes")
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
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems.runtime.backend import vendor_module")
    code.writeline("MAX_GRID_SIZE_X = vendor_module.MAX_GRID_SIZE_X")
    code.writeline("from flag_gems.utils.type_utils import type_promotion")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_functional_repeat_wrapper(
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
        code.writeline("sizes_rank = len(sizes)")
        code.writeline("in0_shape = list(in0.shape)")
        code.writeline("sizes_shape = list(sizes)")
        code.newline()

        code.writeline(
            "assert(sizes_rank >= in0_rank), \
                'Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor'"
        )
        code.writeline("if (sizes_rank > in0_rank): ")
        with code.indent():
            code.writeline("diff = sizes_rank - in0_rank")
            code.writeline("ones = [1 for _ in range(diff)]")
            code.writeline("in0_shape = ones + in0_shape")
        code.newline()
        code.writeline("is_empty = False")
        code.writeline("out_shape = []")
        code.writeline("for i in range(len(in0_shape)): ")
        with code.indent():
            code.writeline(
                "assert(sizes_shape[i] >= 0), 'the number of repetitions per dimension out of range (expected to >= 0) \
                but got {}'.format(sizes_shape[i])"
            )
            code.writeline("if sizes_shape[i] == 0: ")
            with code.indent():
                code.writeline("is_empty = True")
            code.writeline("out_shape.append(in0_shape[i] * sizes_shape[i])")
        code.newline()
        code.writeline(
            "out0 = torch.empty(out_shape, device=in0.device, dtype=in0.dtype)"
        )

        code.writeline("in0 = in0.reshape(in0_shape)")
        code.writeline("if not is_empty: ")
        with code.indent():
            # call destination_passing_func
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


def generate_destination_passing_repeat_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # wrapper signature
    parameters: str = parameter_for_wrapper_out()

    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        # docstring
        if rank > 0:
            code.writeline("shape = out0.shape")
            code.writeline("num_tasks = volume(shape)")

        if rank > 0:
            code.writeline("tile_size = min(512, triton.next_power_of_2(num_tasks))")
            code.writeline("num_warps = 4")
            code.writeline(
                "num_ctas = min(MAX_GRID_SIZE_X//num_warps, triton.cdiv(num_tasks, tile_size))"
            )
            code.writeline(
                "tiles_per_cta = triton.cdiv(num_tasks, tile_size * num_ctas)"
            )
        else:
            code.writeline("num_warps = 1")
            code.writeline("num_ctas = 1")
        code.writeline("grid = (num_ctas, 1, 1)")
        code.newline()

        # input strides for each input tensor w.r.t. the task index space
        if rank > 0:
            code.writeline("# strides of each tensor argument w.r.t the task space")
            code.writeline("in0_strides = in0.stride()")
            code.writeline("in0_shape = in0.shape")
            code.writeline("out0_strides = out0.stride()")
        code.newline()

        # grid
        code.writeline("# kernel launch")

        # launch kernel
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
                    f"{in_shape_args}, # task indexing space used when input and ouput tensor has different shape"
                )
                code.writeline("num_tasks, # num tasks")
                code.writeline("tiles_per_cta=tiles_per_cta, # tiles_per_cta")
                code.writeline("tile_size=tile_size,")
                code.writeline("one_tile_per_cta=tiles_per_cta==1,")
            code.writeline("num_warps=num_warps,")
        code.writeline(")")

        # return
        code.writeline("return out0")
        code.newline()
        code.newline()
    return code


def generate_repeat_kernel(
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

        # signature: strides, for each tensor arguments
        # only add this arguments when rank > 0
        if rank > 0:
            # strides for inputs
            stride_args = ", ".join(f"in0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # strides for in0")

            # strides for outputs
            stride_args = ", ".join(f"out0_stride{j}: int" for j in range(rank))
            code.writeline(f"{stride_args}, # strides for out0")

            # task space, used to reconstruct multi index
            task_space_args = ", ".join(f"s{i}: int" for i in range(rank))
            code.writeline(f"{task_space_args}, # task_space")

            task_space_args2 = ", ".join(f"in_s{i}: int" for i in range(rank))
            code.writeline(
                f"{task_space_args2}, # task_space2 used when input and output tensor has different shape"
            )

            # number of tasks, used to compute mask
            code.writeline("num_tasks: int,")

        # tile size & tiles_per_cta, gsl style
        if rank > 0:
            code.writeline("tiles_per_cta,")

            code.writeline("tile_size: tl.constexpr,")

            code.writeline("one_tile_per_cta: tl.constexpr,")
    code.writeline("):")

    with code.indent():
        # get pid
        code.writeline("# task id & masking")
        pid_stmt = "pid = tl.program_id(0)"
        code.writeline(pid_stmt)

        code.writeline("num_ctas = tl.num_programs(0)")

        # get tid (a.k.a task id)
        tid_stmt = "init_tid = pid * tile_size + tl.arange(0, tile_size)"
        code.writeline(tid_stmt)

        # one-tile-per-cta, monolithic kernel style
        code.writeline("if one_tile_per_cta: # monolitic kernel style")
        with code.indent():
            tid_stmt = "tid = init_tid"
            code.writeline(tid_stmt)

            # only apply masking when rank > 0
            # since we only load a value instead of a block of values when the rank is 0
            mask_stmt: str = "mask = tid < num_tasks"
            code.writeline(mask_stmt)
            code.newline()

            # reconstruct multi index
            code.writeline("# multi index recontruction")
            for i in reversed(range(rank)):
                if i > 0:
                    code.writeline(f"i{i} = tid % s{i}")
                    code.writeline(f"tid //= s{i}")
                else:
                    code.writeline(f"i{i} = tid")
            code.newline()

            # loads
            code.writeline("# loads")
            ptrs_expr: str = " + ".join(
                f"(i{j} % in_s{j}) * in{i}_stride{j}" for j in range(rank)
            )
            ptrs_expr: str = f"in0_ptr + {ptrs_expr}"
            load_stmt: str = f"in0 = tl.load({ptrs_expr}, mask=mask)"
            code.writeline(load_stmt)
            code.newline()

            # compute
            code.writeline("# compute")
            code.writeline("out0 = in0")
            code.newline()

            # stores
            code.writeline("# stores")
            ptrs_expr: str = " + ".join(f"i{j} * out0_stride{j}" for j in range(rank))
            ptrs_expr: str = f"out0_ptr + {ptrs_expr}"
            store_stmt: str = f"tl.store({ptrs_expr}, out0, mask=mask)"
            code.writeline(store_stmt)

        # https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
        code.writeline("else: # grid-stride-loop style kernel")
        with code.indent():
            code.writeline("for j in range(0, tiles_per_cta):")
            with code.indent():
                tid_stmt = "tid = init_tid + j * tile_size * num_ctas"
                code.writeline(tid_stmt)

                # only apply masking when rank > 0
                # since we only load a value instead of a block of values when the rank is 0
                mask_stmt: str = "mask = tid < num_tasks"
                code.writeline(mask_stmt)
                code.newline()

                # reconstruct multi index
                code.writeline("# multi index recontruction")
                for i in reversed(range(rank)):
                    if i > 0:
                        code.writeline(f"i{i} = tid % s{i}")
                        code.writeline(f"tid //= s{i}")
                    else:
                        code.writeline(f"i{i} = tid")
                code.newline()

                # loads
                code.writeline("# loads")
                ptrs_expr: str = " + ".join(
                    f"(i{j} % in_s{j}) * in{i}_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"in0_ptr + {ptrs_expr}"
                load_stmt: str = f"in0 = tl.load({ptrs_expr}, mask=mask)"
                code.writeline(load_stmt)
                code.newline()

                # compute
                code.writeline("# compute")
                code.writeline("out0 = in0")
                code.newline()

                # stores
                code.writeline("# stores")
                ptrs_expr: str = " + ".join(
                    f"i{j} * out0_stride{j}" for j in range(rank)
                )
                ptrs_expr: str = f"out0_ptr + {ptrs_expr}"
                store_stmt: str = f"tl.store({ptrs_expr}, out0, mask=mask)"
                code.writeline(store_stmt)
                code.newline()
    return code


def generate_code(
    rank: int,
    wrapper_name: str,
    destination_passing_func_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # the only runtime determined factor is the rank of the task space
    code = generate_imports(code)
    code = generate_functional_repeat_wrapper(
        wrapper_name, destination_passing_func_name, code
    )
    code = generate_destination_passing_repeat_wrapper(
        rank, destination_passing_func_name, kernel_name, code
    )
    code = generate_repeat_kernel(rank, kernel_name, code)
    return code


class RepeatFunction:
    def __init__(self):
        self.pid = os.getpid()
        # instantiated & cached overloads
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, x, sizes):
        # note: kwargs should not be used in JITFunction directly
        ndim = self.arg_key(x, sizes)
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
                "_repeat_flaggems_jit_function",
                code,
            )

            file_name = f"repeat_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            # do not expose it to sys.modules
            # sys.modules["_add_module"] = m
            spec.loader.exec_module(m)
            overload = getattr(m, "_wrapper")
            self.overloads[key] = overload
        return overload(x, sizes)

    def arg_key(self, x, sizes):
        max_rank = max(x.ndim, len(sizes))
        return max_rank


_repeat_func = RepeatFunction()


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 2**n}, num_stages=3) for n in range(10, 17, 2)
    ],
    key=["C"],
)
@triton.jit
def repeat_2d_kernel(
    inp_ptr,
    out_ptr,
    N,
    C: tl.constexpr,
    repeat_N: tl.constexpr,
    repeat_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    job_id = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    for batch_idx in range(job_id, N, num_jobs):
        if C <= BLOCK_C:
            offset_c = tl.arange(0, C)
            inp_ptrs = inp_ptr + batch_idx * C + offset_c
            inp = tl.load(inp_ptrs).reshape(1, C)
            repeat_inp = inp.broadcast_to(repeat_C, C).reshape(repeat_C * C)
            out_offset_c = tl.arange(0, repeat_C * C)
            for n_idx in range(0, repeat_N):
                out_ptrs = (
                    out_ptr
                    + N * n_idx * repeat_C * C
                    + batch_idx * repeat_C * C
                    + out_offset_c
                )
                tl.store(out_ptrs, repeat_inp)
        else:
            for off in range(0, C, BLOCK_C):
                offset_c = off + tl.arange(0, BLOCK_C)
                inp_ptrs = inp_ptr + batch_idx * C + offset_c
                inp_mask = offset_c < C
                inp = tl.load(inp_ptrs, mask=inp_mask, other=0)
                for c_idx in range(0, repeat_C):
                    for n_idx in range(0, repeat_N):
                        out_ptrs = (
                            out_ptr
                            + N * n_idx * repeat_C * C
                            + batch_idx * repeat_C * C
                            + c_idx * C
                            + offset_c
                        )
                        tl.store(out_ptrs, inp, mask=inp_mask)


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON REPEAT")

    inp_rank = inp.dim()
    sizes_rank = len(sizes)
    if inp_rank == 2 and sizes_rank == 2:
        inp_shape = list(inp.shape)
        sizes_shape = list(sizes)
        N = inp_shape[0]
        C = inp_shape[1]
        repeat_N = sizes_shape[0]
        repeat_C = sizes_shape[1]

        is_empty = False
        out_shape = []
        for i in range(len(inp_shape)):
            assert sizes_shape[i] >= 0
            if sizes_shape[i] == 0:
                is_empty = True
            out_shape.append(inp_shape[i] * sizes_shape[i])
        out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
        if is_empty:
            return out
        repeat_2d_kernel[(TOTAL_CORE_NUM,)](
            inp.contiguous(), out, N, C, repeat_N, repeat_C
        )
        return out

    out = _repeat_func(inp, sizes)
    return out
