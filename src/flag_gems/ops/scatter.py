import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, NameSpace


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.newline()
    code.newline()
    return code


def generate_scatter_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the autotune function
    code.writeline("def cfggen():")
    with code.indent():
        code.writeline("block_m = [1, 2, 4, 8]")
        code.writeline("block_n = [256, 512, 1024, 2048]")
        code.writeline("configs = [")
        with code.indent():
            code.writeline('triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)')
            code.writeline("for m in block_m")
            code.writeline("for n in block_n")
        code.writeline("]")
        code.writeline("return configs")

    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline('@triton.autotune(configs=cfggen(), key=["M", "N"])')
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    function_ns = NameSpace()
    with code.indent():
        if rank > 0:
            code.writeline("src_strided,")
            function_ns.create_name("src_strided")
            code.writeline("index,")
            function_ns.create_name("index")
            code.writeline("inp,")
            function_ns.create_name("inp")
            code.writeline("out,")
            function_ns.create_name("out")

            for i in range(rank):
                function_ns.create_name(f"inp_stride_{i}")
            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            for i in range(rank):
                function_ns.create_name(f"index_stride_{i}")
            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            for i in range(rank):
                function_ns.create_name(f"index_shape_{i}")
            shape_args = ", ".join(f"index_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for index")

            code.writeline("dim,")
            code.writeline("stride_dim,")
            code.writeline("M,")
            code.writeline("N,")
            # reduce options
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK_M: tl.constexpr,")
            code.writeline("BLOCK_N: tl.constexpr,")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid_x = tl.program_id(0)")
        code.writeline("pid_y = tl.program_id(1)")
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

        #   1. Calculate inp_offsets and idx_offsets
        code.writeline("inp_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("idx_offsets = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int64)")
        code.writeline("cur_idx = rows_offsets * N + cols_offsets")

        #   2. snippets
        for i in range(rank):
            code.writeline(f"mod = cur_idx % index_shape_{i}")
            code.writeline(f"inp_offsets += mod * inp_stride_{i}")
            code.writeline(f"idx_offsets += mod * index_stride_{i}")
            if i != (rank - 1):
                code.writeline(f"cur_idx = cur_idx // index_shape_{i}")

        #   3. Use offsets to scatter
        code.writeline(
            "cur_src = tl.load(src_strided + idx_offsets, mask=mask, other=0)"
        )
        code.writeline("cur_index = tl.load(index + idx_offsets, mask=mask, other=0)")
        code.writeline("inp_offsets += cur_index * stride_dim")

        code.newline()
        code.writeline("if IS_ADD: ")
        with code.indent():
            code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
            code.writeline("res = cur_inp + cur_src")
            code.writeline("tl.store(out + inp_offsets, res, mask=mask)")

        code.writeline("elif IS_MUL: ")
        with code.indent():
            code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
            code.writeline("res = cur_inp * cur_src")
            code.writeline("tl.store(out + inp_offsets, res, mask=mask)")

        code.writeline("else: ")
        with code.indent():
            code.writeline("tl.store(out + inp_offsets, cur_src, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # src_strided, index, inp, out, dim, M, N, reduce
    parameters: List[str] = []

    parameters.append("src_strided")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim")
    parameters.append("M")
    parameters.append("N")
    parameters.append("reduce")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name}({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(inp.stride())")
        code.writeline("index_strides = index.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("stride_dim = inp_strides[dim]")
        code.writeline("inp_strides[dim] = 0")

        code.writeline('IS_ADD = reduce == "add"')
        code.writeline('IS_MUL = reduce == "multiply"')

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(M, meta["BLOCK_M"]),')
            code.writeline('triton.cdiv(N, meta["BLOCK_N"])')
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

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("dim,")
                code.writeline("stride_dim,")
                code.writeline("M,")
                code.writeline("N,")
                # reduce options
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src_strided, index, inp, out, dim, M, N, reduce]
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
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
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
            )

            file_name = f"scatter_rank_{key}_pid_{self.pid}.py"

            with open(cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_scatter_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def scatter(inp, dim, index, src, reduce=None):
    logging.debug("GEMS SCATTER")
    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()
    out = inp.clone()

    src_strided = src.as_strided(index.shape, src.stride()).contiguous()
    # plain_idx = torch.arange(0, index.numel(), device=inp.device).reshape(index.shape)
    N = list(index.shape)[index.ndim - 1]
    M = index.numel() // N

    _scatter_func(src_strided, index, inp, out, dim, M, N, reduce)
    return out
