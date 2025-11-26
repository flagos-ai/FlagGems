import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code


def generate_index_fill_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("inp_numel,")
            code.writeline("inp_stride_dim,")
            code.writeline("inp_shape_dim,")
            code.writeline("value,")

            stride_args = ", ".join(f"inp_stride_{i}: int" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for input")

            shape_args = ", ".join(f"inp_shape_{i}: int" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for input")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

        # Kernel Code
        with code.indent():
            code.writeline("pid = tl.program_id(axis=0)")
            code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < N")

            # Calculate multi-dimensional indices
            for i in range(rank - 1, -1, -1):
                code.writeline(f"inp_offset{i} = offsets % inp_shape_{i}")
                code.writeline(f"offsets = offsets // inp_shape_{i}")
            code.newline()
            
            # Calculate linear offset in the input tensor
            comp = [f"inp_offset{i} * inp_stride_{i}" for i in range(rank)]
            code.writeline(f"inp_offset = {' + '.join(comp)}")

            # Calculate index for the dimension being filled
            code.writeline("pre_cal = inp_stride_dim * inp_shape_dim")
            code.writeline("pre_idx = (inp_offset // pre_cal).to(tl.int64)")
            code.writeline(
                "dim_idx = (inp_offset % pre_cal // inp_stride_dim).to(tl.int64)"
            )
            
            # Load index values and validate
            code.writeline(
                "src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)"
            )
            code.writeline(
                'assert src_dim_idx >= 0 and src_dim_idx < inp_shape_dim, "0 <= index < self.size(dim)"'
            )
            
            # Calculate the final index in the output tensor
            code.writeline(
                "output_idx = (inp_offset + (src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)"
            )
            code.writeline("output_mask = output_idx < inp_numel")
            
            # Fill the value at the calculated positions
            code.writeline("tl.store(out + output_idx, value, mask=output_mask)")

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, value, dim, inp_stride_dim, inp_shape_dim, N, inp.numel()
    parameters: List[str] = []
    parameters.append("out")
    parameters.append("index")
    parameters.append("value")
    parameters.append("dim")
    parameters.append("inp_stride_dim")
    parameters.append("inp_shape_dim")
    parameters.append("N")
    parameters.append("inp_numel")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name} ({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("inp_strides = list(out.stride())")
        code.writeline("inp_shapes = list(out.shape)")

        # kernel launch
        code.writeline("BLOCK_SIZE = 128")  # BLOCK_SIZE setting
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, out, N, inp_numel, inp_stride_dim, inp_shape_dim, value, "
            )
            if rank > 0:
                s = ", ".join(f"inp_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"inp_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, value, dim, inp_stride_dim, inp_shape_dim, N, inp.numel()]
    shape = inputs[0].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_index_fill_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexFillFunction:
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
                "_index_fill_wrapper",
                "_index_fill_jit_function",
                code,
            )

            file_name = f"index_fill_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_fill_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        tensors = [item for item in args if torch.is_tensor(item)]
        max_rank = max(item.ndim for item in tensors)
        return max_rank


_index_fill_func = IndexFillFunction()


def index_fill(inp, dim, index, value):
    logger.debug("GEMS INDEX FILL")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    out = inp.clone()

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    inp_shape_dim = inp.size(dim)
    N = out.numel()

    _index_fill_func(
        out,
        index,
        value,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        N,
        inp.numel(),
    )
    return out


def index_fill_(inp, dim, index, value):
    logger.debug("GEMS INDEX FILL_")
    assert ((0 <= index) * (index < inp.size(dim))).equal(
        torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    ), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    inp_shape_dim = inp.size(dim)
    N = inp.numel()

    _index_fill_func(
        inp,
        index,
        value,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        N,
        inp.numel(),
    )
    return inp