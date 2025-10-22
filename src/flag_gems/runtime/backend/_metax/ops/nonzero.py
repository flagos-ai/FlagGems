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
    code.writeline("from flag_gems.utils import libentry, libtuner")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.runtime import torch_device_fn")

    code.newline()
    code.newline()

    return code


def generate_nonzero_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # the decorators
    code.writeline("@libentry()")
    code.writeline("@libtuner(")
    with code.indent():
        code.writeline("configs=runtime.get_tuned_config('nonzero'),")
        code.writeline("key=['n_elements',],)")
    code.writeline("@triton.jit")

    # signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("inp,")
            code.writeline("prefix_sum,")
            code.writeline("out,")
            code.writeline("n_elements: tl.constexpr,")
            code.writeline("ndim: tl.constexpr,")

            shape_args = ", ".join(f"dim{i}_size" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

        # Kernel Code
        with code.indent():
            code.writeline("pid = tle.program_id(0)")
            code.writeline("offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offset < n_elements")
            code.newline()

            code.writeline("inp_vals = tl.load(inp + offset, mask=mask)")
            code.writeline("out_offset = tl.load(prefix_sum + offset, mask=mask) - 1")
            code.writeline("nonzero_mask = mask and inp_vals == True  # noqa")
            code.writeline("idx_flat = offset")
            code.newline()

            for i in range(rank - 1, -1, -1):
                code.writeline(f"remainder = idx_flat % dim{i}_size")
                code.writeline(f"idx_flat //= dim{i}_size")
                code.writeline(
                    f"tl.store(out + out_offset * ndim + {i}, remainder, mask=nonzero_mask)"
                )
                code.newline()

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # inp_bool, prefix_sum, out, n_elements, inp_ndim, shape
    parameters: List[str] = []
    parameters.append("inp_bool")
    parameters.append("prefix_sum")
    parameters.append("out")
    parameters.append("n_elements")
    parameters.append("inp_ndim")
    parameters.append("shape")

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
        code.writeline(
            'grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)'
        )
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline("inp_bool, prefix_sum, out, n_elements, inp_ndim,  ")
            if rank > 0:
                s = ", ".join(f"shape[{i}]" for i in range(rank))
                code.writeline(f"{s}")

        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [inp_bool, prefix_sum, out, n_elements, inp_ndim, shape]
    shape = inputs[-1]
    rank = len(shape)
    code = generate_imports(code)
    code = generate_nonzero_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class NonzeroFunction:
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
                "_nonzero_wrapper",
                "_nonzero_jit_function",
                code,
            )

            file_name = f"nonzero_rank_{key}_pid_{self.pid}.py"

            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_nonzero_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        # args: [inp_bool, prefix_sum, out, n_elements, inp_ndim, shape]
        return args[-2]


_nonzero_func = NonzeroFunction()


def nonzero(inp, *, as_tuple=False):
    logger.debug("METAX GEMS NONZERO")

    assert len(inp.shape) > 0, "Invalid input shape, input dimension must > 0"
    inp_ndim = inp.ndim
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    shape = inp.shape

    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0

    prefix_sum = inp_bool.cumsum(axis=0)

    num_nonzeros = n_elements
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    _nonzero_func(inp_bool, prefix_sum, out, n_elements, inp_ndim, shape)

    num_nonzeros = prefix_sum[n_elements - 1].item()
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
