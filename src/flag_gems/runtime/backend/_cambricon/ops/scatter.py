import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_scatter_kernel(
    rank: int,
    dim: int,
    large_tensor: bool,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the autotune function

    code.newline()
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline(
        '@libtuner(configs=runtime.get_tuned_config("scatter"), key=["N"], strategy=["log"],'
    )
    code.writeline('          restore_value=["out"], )')

    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("src,")
            code.writeline("index,")
            code.writeline("inp,")
            code.writeline("out,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"index_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for index")

            code.writeline("dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            # reduce options
            code.writeline("IS_ADD: tl.constexpr,")
            code.writeline("IS_MUL: tl.constexpr,")
            code.writeline("BLOCK_SIZE: tl.constexpr,")

    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < N")

        #   1. Calculate inp_offsets and src_offsets
        if large_tensor:
            code.writeline("inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)")
            code.writeline("src_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)")
        else:
            code.writeline("inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)")
            code.writeline("src_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)")

        code.writeline("cur_idx = offsets")

        #   2. snippets
        for i in range(rank - 1, -1, -1):
            code.writeline(f"mod = cur_idx % index_shape_{i}")
            if dim != i:
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
            code.writeline(f"src_offsets += mod * src_stride_{i}")
            # the last "//" should be optimized out
            code.writeline(f"cur_idx = cur_idx // index_shape_{i}")

        #   3. Use offsets to scatter
        code.writeline("cur_src = tl.load(src + src_offsets, mask=mask, other=0)")
        if large_tensor:
            code.writeline("cur_index = tl.load(index + offsets, mask=mask, other=0)")
        else:
            code.writeline(
                "cur_index = tl.load(index + offsets, mask=mask, other=0).to(tl.int32)"
            )
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
    # src, index, inp, out, dim, reduce, N
    parameters: List[str] = []

    parameters.append("src")
    parameters.append("index")
    parameters.append("inp")
    parameters.append("out")
    parameters.append("dim")
    parameters.append("reduce")
    parameters.append("N")

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
        code.writeline("src_strides = src.stride()")
        code.writeline("index_shapes = list(index.shape)")
        code.writeline("stride_dim = inp_strides[dim]")
        code.writeline("inp_strides[dim] = 0")

        code.writeline('IS_ADD = reduce == "add"')
        code.writeline('IS_MUL = reduce == "multiply"')

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK_SIZE"]),')
        code.writeline(")")

        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)

        with code.indent():
            code.writeline("src, index, inp, out, ")
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"index_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                code.writeline("dim,")
                code.writeline("stride_dim,")
                code.writeline("N,")
                # reduce options
                code.writeline("IS_ADD,")
                code.writeline("IS_MUL,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    rank: int,
    dim: int,
    large_input: bool,
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [src, index, inp, out, dim, reduce, N]
    shape = inputs[1].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_scatter_kernel(rank, dim, large_input, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class ScatterFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        rank = kwargs["rank"]
        dim = kwargs["dim"]
        large_tensor = kwargs["large_tensor"]

        key = f"{self.arg_key(*args)}_{rank}_{dim}_{large_tensor}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                rank,
                dim,
                large_tensor,
                args,
                "_scatter_wrapper",
                "_scatter_jit_function",
                code,
            )

            file_name = f"scatter_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
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

        return overload(*args)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_scatter_func = ScatterFunction()


def scatter(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_CAMBRICON SCATTER")
    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()
    out = inp.clone()

    N = index.numel()

    large_tensor = (src.numel() * src.element_size() > 2**31) or (
        out.numel() * out.element_size() > 2**31
    )

    # <rank>_<dim>_<large_tensor> is part of the key of overloads
    _scatter_func(
        src,
        index,
        inp,
        out,
        dim,
        reduce,
        N,
        rank=len(index.shape),
        large_tensor=large_tensor,
        dim=dim,
    )
    return out


def scatter_(inp, dim, index, src, reduce=None):
    logger.debug("GEMS_CAMBRICON SCATTER_")
    inp = inp.contiguous()
    index = index.contiguous()
    src = src.contiguous()
    out = inp

    N = index.numel()

    large_tensor = (src.numel() * src.element_size() > 2**31) or (
        out.numel() * out.element_size() > 2**31
    )

    _scatter_func(
        src,
        index,
        inp,
        out,
        dim,
        reduce,
        N,
        rank=len(index.shape),
        large_tensor=large_tensor,
        dim=dim,
    )

    return inp
