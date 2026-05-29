import importlib
import logging
import os
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

GCU_MAX_GRID_Y = 255


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as tle")
    code.newline()
    code.newline()
    return code


def generate_index_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["out_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [f"out_stride{i}," for i in range(index_rank + inp_rank - indices_len)]
        args += [
            "M,",
            "N,",
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
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0).to(tl.int32)"
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
        comp = [f"indices_idx{i} * out_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * out_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"out_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(input_ptr + input_offset, mask=mask)")
        code.writeline("tl.store(out_ptr + out_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(f"def {wrapper_name}(input, indices, out):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("out_shape = out.shape")
        code.writeline("out_stride = out.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        if inp_rank == indices_len:
            code.writeline("BLOCK_SIZE0 = min(_next_pow2(M), 128)")
            code.writeline("BLOCK_SIZE1 = 1")
        else:
            code.writeline("BLOCK_SIZE0 = min(_next_pow2(M), 4)")
            code.writeline(
                "BLOCK_SIZE1 = max(_next_pow2(max(triton.cdiv(N, 255), 1)), min(_next_pow2(N), 256))"
            )
            code.writeline("BLOCK_SIZE1 = min(BLOCK_SIZE1, 2048)")
        code.newline()
        code.writeline("grid = (")
        with code.indent():
            code.writeline("triton.cdiv(M, BLOCK_SIZE0),")
            code.writeline("triton.cdiv(N, BLOCK_SIZE1),")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["out,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"out_stride[{i}]," for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,"]
            args += ["BLOCK_SIZE0=BLOCK_SIZE0,", "BLOCK_SIZE1=BLOCK_SIZE1,"]
            args += ["num_warps=4,"]
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
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    code.newline()
    code.writeline("def _next_pow2(n):")
    with code.indent():
        code.writeline("if n <= 1: return 1")
        code.writeline("return 1 << (n - 1).bit_length()")
    code.newline()
    code.newline()
    generate_index_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, out = args
        full_args = (inp, tensor_indices)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_index_wrapper",
                "_index_jit_function",
                code,
            )

            file_name = f"index_gcu400_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_gcu400_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices = args[0], args[1]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_func = IndexFunction()


def index(inp, indices):
    logger.debug("GEMS INDEX GCU400")
    original_indices = list(indices)
    indices = list(indices)

    if not indices:
        raise ValueError("at least one index must be provided")

    indices = [
        index.to(inp.device)
        if index is not None and index.device != inp.device
        else index
        for index in indices
    ]

    processed_indices = []
    for i, index_t in enumerate(indices):
        if index_t is not None:
            if index_t.dtype in [torch.int8, torch.bool]:
                nonzero = index_t.nonzero()
                k = len(processed_indices)
                if k + index_t.ndim > inp.ndim:
                    raise IndexError(
                        f"too many indices for tensor of dimension {inp.ndim}"
                    )
                for j in range(index_t.ndim):
                    if index_t.shape[j] != inp.shape[k + j]:
                        raise IndexError(
                            f"The shape of the mask {index_t.shape} at index {i} "
                            f"does not match the shape of the indexed tensor {inp.shape} at index {k + j}"
                        )
                for j in range(index_t.ndim):
                    processed_indices.append(nonzero.select(1, j))
            elif index_t.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                processed_indices.append(index_t)
            else:
                raise TypeError(
                    "tensors used as indices must be long, int, byte or bool tensors"
                )
        else:
            processed_indices.append(None)

    indices = processed_indices

    if len(indices) > inp.ndim:
        raise IndexError(
            f"too many indices for tensor of dimension {inp.ndim} (got {len(indices)})"
        )

    has_any_tensor = any(idx is not None for idx in indices)
    starts_with_none = indices[0] is None if indices else False

    tensor_indices = [idx for idx in indices if idx is not None]
    if tensor_indices:
        if len(tensor_indices) > 1:
            tensor_indices = list(torch.broadcast_tensors(*tensor_indices))
        tensor_idx = 0
        for i in range(len(indices)):
            if indices[i] is not None:
                indices[i] = tensor_indices[tensor_idx]
                tensor_idx += 1

    while len(indices) < inp.ndim:
        indices.append(None)

    state = 0
    has_contiguous_subspace = False
    for idx_item in indices:
        if state == 0:
            if idx_item is not None:
                state = 1
        elif state == 1:
            if idx_item is None:
                state = 2
        else:
            if idx_item is not None:
                break
    else:
        has_contiguous_subspace = True

    need_post_process = False
    first_tensor_dim = None
    if not has_contiguous_subspace or (starts_with_none and has_any_tensor):
        dims = []
        transposed_indices = []
        for i, idx_item in enumerate(indices):
            if idx_item is not None:
                dims.append(i)
                transposed_indices.append(idx_item)
        for i, idx_item in enumerate(indices):
            if idx_item is None:
                dims.append(i)
                transposed_indices.append(idx_item)
        inp = inp.permute(dims)
        indices = transposed_indices

        if starts_with_none and has_any_tensor and has_contiguous_subspace:
            need_post_process = True
            for i, idx in enumerate(original_indices):
                if idx is not None:
                    first_tensor_dim = i
                    break

    before_shape = []
    after_shape = []
    replacement_shape = []

    for dim, idx_item in enumerate(indices):
        if idx_item is None:
            if replacement_shape:
                after_shape.append(inp.shape[dim])
            else:
                before_shape.append(inp.shape[dim])
        else:
            if not replacement_shape:
                replacement_shape = list(idx_item.shape)

    out_shape = before_shape + replacement_shape + after_shape

    # GCU fix: handle int64 by computing in int32
    original_dtype = inp.dtype
    need_int64_cast = original_dtype == torch.int64
    if need_int64_cast:
        inp = inp.to(torch.int32)

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    if inp.numel() == 0:
        if need_int64_cast:
            return out.to(torch.int64)
        return out

    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        result = inp.view(*out_shape)
        if need_int64_cast:
            result = result.to(torch.int64)
        return result

    # GCU fix: convert int64 indices to int32
    tensor_indices = [
        idx.to(torch.int32) if idx.dtype == torch.int64 else idx
        for idx in tensor_indices
    ]

    _index_func(inp, tensor_indices, out)

    if need_post_process:
        index_rank = tensor_indices[0].ndim
        pre_dims = list(range(index_rank, index_rank + first_tensor_dim))
        broadcast_dims = list(range(index_rank))
        post_dims = list(range(index_rank + first_tensor_dim, out.ndim))
        new_order = pre_dims + broadcast_dims + post_dims
        out = out.permute(new_order)

    if need_int64_cast:
        out = out.to(torch.int64)

    return out
