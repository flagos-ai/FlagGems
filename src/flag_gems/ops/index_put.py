import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    max_rank = max([len(index.shape) for index in indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")

    code.newline()
    code.newline()
    return code


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline(
        '@triton.autotune(configs=runtime.get_tuned_config("index_put"), key=["M", "N"], restore_value=["input_ptr"])'
    )
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [
            f"values_stride{i}," for i in range(index_rank + inp_rank - indices_len)
        ]
        args += [
            "M,",
            "N,",
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK_SIZE0: tl.constexpr,",
            "BLOCK_SIZE1: tl.constexpr,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = tle.program_id(axis=0)")
        code.writeline("pid1 = tle.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        else:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
        code.newline()
        code.writeline("cur_idx = offset1")
        for i in range(inp_rank - 1, indices_len - 1, -1):
            code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
            code.writeline(f"cur_idx = cur_idx // input_shape{i}")
        code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
            )
        code.newline()
        index_mask = [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
            for i in range(indices_len)
        ]
        code.writeline(f"index_mask = {' & '.join(index_mask)}")
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = index_mask & mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
        comp += [
            f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)
        ]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"indices_idx{i} * values_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * values_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"values_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(values_ptr + values_offset, mask=mask)")
        code.writeline("if IS_ACCUMULATE:")
        with code.indent():
            code.writeline(
                "tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask)"
            )
        code.writeline("else:")
        with code.indent():
            code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_put_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(f"def {wrapper_name}(input, indices, values, accumulate):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_shape = values.shape")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']), ")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']), ")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"values_stride[{i}],"
                for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,", "accumulate==True,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp_rank = inputs[0].ndim
    indices_len = len(inputs[1])
    index_rank = inputs[1][0].ndim
    code = generate_imports(code)
    generate_index_put_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_put_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexPutFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = self.arg_key(*args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                args,
                "_index_put_wrapper",
                "_index_put_jit_function",
                code,
            )
            file_name = f"index_put_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_put_wrapper")
            self.overloads[key] = overload

        return overload(*args, **kwargs)

    def arg_key(self, *args):
        inp_rank = args[0].ndim
        indices_len = len(args[1])
        index_rank = args[1][0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_put_func = IndexPutFunction()


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX PUT")

    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()

        if values.numel() == 1:
            values = torch.full((K,), values.item(), dtype=inp.dtype, device=inp.device)
        elif values.numel() == K:
            values = values.reshape((K,))

    indices = [
        index.to(inp.device) if index.device != inp.device else index
        for index in indices
    ]
    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    out = inp.clone()
    _index_put_func(out, indices, values, accumulate)
    return out


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX PUT_")

    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()

        if values.numel() == 1:
            values = torch.full((K,), values.item(), dtype=inp.dtype, device=inp.device)
        elif values.numel() == K:
            values = values.reshape((K,))

    indices = [
        index.to(inp.device) if index.device != inp.device else index
        for index in indices
    ]
    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    _index_put_func(inp, indices, values, accumulate)
    return inp
