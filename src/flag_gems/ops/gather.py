import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.ops.scatter import scatter_
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import torch")
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")

    code.newline()
    code.newline()
    return code


def generate_gather_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # make the inlined function visible in the context
    code.newline()

    code.writeline("@libentry()")
    code.writeline("@triton.heuristics({'BLOCK_SIZE_N': lambda args: 512})")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = [
            "inp, ",
            "index, ",
            "out, ",
        ]
        args += [f"inp_shape{i}," for i in range(rank)]
        args += [f"index_shape{i}, " for i in range(rank)]
        args += [f"out_shape{i}, " for i in range(rank)]
        args += [f"inp_stride{i}, " for i in range(rank)]
        args += [f"index_stride{i}, " for i in range(rank)]
        args += [f"out_stride{i}, " for i in range(rank)]
        args += ["dim, ", "dim_stride, ", "N, ", "BLOCK_SIZE_N: tl.constexpr, "]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid = tle.program_id(0)")
        code.writeline(
            "offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)"
        )
        code.newline()
        code.writeline("cur_offset = offset")
        for i in range(rank - 1, -1, -1):
            code.writeline(f"index_idx{i} = cur_offset % index_shape{i}")
            code.writeline(f"cur_offset = cur_offset // index_shape{i}")
        code.newline()
        comp = [f"index_idx{i} * index_stride{i}" for i in range(rank)]
        code.writeline(f"index_offset = {' + '.join(comp)}")
        code.writeline("mask = offset < N")
        code.writeline("cur_index = tl.load(index + index_offset, mask=mask, other=0)")
        code.newline()
        comp = [f"index_idx{i} * inp_stride{i}" for i in range(rank)]
        code.writeline(f"inp_offset = {' + '.join(comp)}")
        code.writeline("inp_offset += cur_index * dim_stride")
        code.writeline("cur_inp = tl.load(inp + inp_offset, mask=mask, other=0)")
        code.newline()
        comp = [f"index_idx{i} * out_stride{i}" for i in range(rank)]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        code.writeline("tl.store(out + out_offset, value=cur_inp, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_gather_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    code.writeline(f"def {wrapper_name}(inp, dim, index, out, dim_stride, N):")
    with code.indent():
        code.writeline("inp_shape = inp.shape")
        code.writeline("inp_stride = inp.stride()")
        code.writeline("index_shape = index.shape")
        code.writeline("index_stride = index.stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline("grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )")
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = [
                "inp, ",
                "index, ",
                "out, ",
            ]
            args += [f"inp_shape[{i}], " for i in range(rank)]
            args += [f"index_shape[{i}], " for i in range(rank)]
            args += [f"out_shape[{i}], " for i in range(rank)]
            args += [f"inp_stride[{i}], " for i in range(rank)]
            args += [f"index_stride[{i}], " for i in range(rank)]
            args += [f"out_stride[{i}], " for i in range(rank)]
            args += [
                "dim, ",
                "dim_stride, ",
                "N, ",
            ]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return out")
    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    rank = inputs[0].ndim

    code = generate_imports(code)
    code = generate_gather_kernel(rank, kernel_name, code)
    code = generate_gather_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class GatherFunction:
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
                "_gather_wrapper",
                "_gather_flaggems_jit_function",
                code,
            )

            file_name = f"gather_rank_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_gather_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        return args[0].ndim


_gather_func = GatherFunction()


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS GATHER")
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    N = index.numel()
    _gather_func(inp_strided, dim, index, out, dim_stride, N)
    return out


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS GATHER BACKWARD")
    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
