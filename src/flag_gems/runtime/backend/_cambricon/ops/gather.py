import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer
from flag_gems.utils.shape_utils import restride_dim

from .scatter import scatter

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


def generate_gather_kernel(
    dim: int,
    large_input: bool,
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    # the decorators
    code.writeline("@libentry()")
    code.writeline(
        '@libtuner(configs=runtime.get_tuned_config("gather"), key=["N"], strategy=["log"])'
    )
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("inp,")
            code.writeline("out,")
            code.writeline("index,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for inp")

            stride_args = ", ".join(f"index_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for index")

            shape_args = ", ".join(f"index_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for index")

            code.writeline("dim,")
            code.writeline("stride_dim,")
            code.writeline("N,")
            code.writeline("BLOCK_SIZE: tl.constexpr,")
    code.writeline("):")

    # Kernel Code
    with code.indent():
        code.writeline("pid = tl.program_id(0)")
        code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        code.writeline("mask = offsets < N")

        #   1. Calculate inp_offsets and idx_offsets
        if large_input:
            code.writeline("inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)")
        else:
            code.writeline("inp_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)")
        code.writeline("index_offsets = offsets")

        #   2. snippets
        for i in range(rank - 1, -1, -1):
            if not (dim == 0 and i == 0):
                code.writeline(f"mod = offsets % index_shape_{i}")

            if i != dim:
                # will be corrected by adding cur_index*stride_dim
                code.writeline(f"inp_offsets += mod * inp_stride_{i}")
            if i != 0:
                code.writeline(f"offsets //= index_shape_{i}")

        # Use offsets to gather
        if large_input:
            code.writeline(
                "cur_index = tl.load(index + index_offsets, mask=mask, other=0)"
            )
        else:
            code.writeline(
                "cur_index = tl.load(index + index_offsets, mask=mask, other=0).to(tl.int32)"
            )

        code.writeline("inp_offsets += cur_index * stride_dim")

        code.writeline("cur_inp = tl.load(inp + inp_offsets, mask=mask, other=0)")
        code.writeline("tl.store(out + index_offsets, cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def parameter_for_wrapper() -> str:
    # inp_strided, out, index, dim, stride_dim, N
    parameters: List[str] = []

    parameters.append("inp_strided")
    parameters.append("out")
    parameters.append("index")
    parameters.append("dim")
    parameters.append("stride_dim")
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
        code.writeline("inp_strides = inp_strided.stride()")
        code.writeline("index_strides = index.stride()")
        code.writeline("index_shapes = list(index.shape)")

        # kernel launch
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline('triton.cdiv(N, meta["BLOCK_SIZE"]),')
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
                code.writeline("N,")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    dim: int,
    large_input: bool,
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: inp_strided, out, index, dim, stride_dim, N, large_input
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_gather_kernel(dim, large_input, rank, kernel_name, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        rank = kwargs["rank"]
        dim = kwargs["dim"]
        large_input = kwargs["large_input"]

        key = f"{self.arg_key(*args)}_{rank}_{dim}_{large_input}"
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                dim,
                large_input,
                args,
                "_gather_wrapper",
                "_gather_jit_function",
                code,
            )

            file_name = f"gather_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
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

        return overload(*args)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_gather_func = GatherFunction()


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_CAMBRICON GATHER")
    inp = inp.contiguous()
    index = index.contiguous()
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    out = out.contiguous()
    stride_dim = inp.stride(dim)

    inp_strided = restride_dim(inp, dim, index.shape)
    N = index.numel()

    large_input = inp.numel() * inp.element_size() > 2**31
    rank = len(index.shape)

    # <rank>_<dim>_<large_input> is the key of overloads
    # large_input is only for key
    _gather_func(
        inp_strided,
        out,
        index,
        dim,
        stride_dim,
        N,
        large_input=large_input,
        dim=dim,
        rank=rank,
    )
    return out


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_CAMBRICON GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter(result, dim, index, grad, reduce="add")
