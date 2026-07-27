# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    # Filter out None values (basic indexing markers)
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = max([len(index.shape) for index in tensor_indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if index is not None and tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")

    code.newline()
    code.newline()
    return code


def _gen_kernel_inner_body(inp_rank, indices_len, index_rank, code):
    code.newline()
    code.writeline("cur_idx_n = offset1")
    for i in range(inp_rank - 1, indices_len - 1, -1):
        code.writeline(f"input_idx{i} = cur_idx_n % input_shape{i}")
        code.writeline(f"cur_idx_n = cur_idx_n // input_shape{i}")
    code.newline()
    code.writeline("mask1 = offset1 < N")
    code.writeline("mask = index_mask & mask0 & mask1")
    code.newline()
    comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
    comp += [f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)]
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
            "cur_input = tl.load(input_ptr + input_offset, mask=mask, other=0.0)"
        )
        code.writeline(
            "tl.store(input_ptr + input_offset, cur_input + cur_value, mask=mask)"
        )
    code.writeline("else:")
    with code.indent():
        code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")


def generate_mmu_heuristics_for_index_put(inp_rank, code: IndentedBuffer):
    stride_items = ", ".join(f"args['input_stride{i}']" for i in range(inp_rank))
    shape_items = ", ".join(f"args['input_shape{i}']" for i in range(inp_rank))

    code.writeline("def _mmu_safe_index_put_blocks(args):")
    with code.indent():
        code.writeline(
            "from _enflame.gcu300.utils.shape_utils import mmu_safe_index_put_block_sizes"
        )
        code.writeline(f"input_stride = ({stride_items},)")
        code.writeline(f"input_shape = ({shape_items},)")
        code.writeline("return mmu_safe_index_put_block_sizes(")
        code.writeline("    args['BLOCK_SIZE0'],")
        code.writeline("    args['BLOCK_SIZE1'],")
        code.writeline("    input_stride,")
        code.writeline("    input_shape,")
        code.writeline("    args['input_ptr'].element_size(),")
        code.writeline(")")
    code.newline()

    code.writeline("def heur_index_put_block_size0(args):")
    with code.indent():
        code.writeline("return _mmu_safe_index_put_blocks(args)[0]")
    code.newline()

    code.writeline("def heur_index_put_block_size1(args):")
    with code.indent():
        code.writeline("return _mmu_safe_index_put_blocks(args)[1]")
    code.newline()
    code.newline()


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline(
        '@triton.autotune(configs=runtime.get_tuned_config("index_put"), key=["M", "N"])'
    )
    code.writeline("@triton.heuristics(")
    with code.indent():
        code.writeline("values={")
        with code.indent():
            code.writeline('"BLOCK_SIZE0": heur_index_put_block_size0,')
            code.writeline('"BLOCK_SIZE1": heur_index_put_block_size1,')
        code.writeline("},")
    code.writeline(")")
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
        code.writeline("pid0 = tl.program_id(axis=0)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
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
        code.newline()
        if inp_rank == indices_len:
            code.writeline("pid1 = tl.program_id(axis=1)")
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
            _gen_kernel_inner_body(inp_rank, indices_len, index_rank, code)
        else:
            code.writeline("num_pid1 = tl.num_programs(1)")
            code.writeline("num_blocks_n = (N + BLOCK_SIZE1 - 1) // BLOCK_SIZE1")
            code.writeline(
                "for pid1 in tl.range(tl.program_id(1), num_blocks_n, num_pid1):"
            )
            with code.indent():
                code.writeline(
                    "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
                )
                _gen_kernel_inner_body(inp_rank, indices_len, index_rank, code)

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
        code.writeline("element_size = input.element_size()")
        code.newline()
        code.writeline("def grid(meta):")
        with code.indent():
            code.writeline("block_size0, block_size1 = mmu_safe_index_put_block_sizes(")
            code.writeline("    meta['BLOCK_SIZE0'],")
            code.writeline("    meta['BLOCK_SIZE1'],")
            code.writeline("    input_stride,")
            code.writeline("    input_shape,")
            code.writeline("    element_size,")
            code.writeline(")")
            code.writeline("return (")
            with code.indent():
                code.writeline("triton.cdiv(M, block_size0),")
                if inp_rank != indices_len:
                    code.writeline("min(triton.cdiv(N, block_size1), 255),")
                else:
                    code.writeline("triton.cdiv(N, block_size1),")
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
    code.writeline(
        "from _enflame.gcu300.utils.shape_utils import mmu_safe_index_put_block_sizes"
    )
    code.newline()
    generate_mmu_heuristics_for_index_put(inp_rank, code)
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
    logger.debug("GEMS_ENFLAME INDEX_PUT")

    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()
        target_shape = (K,) + inp.shape[len(indices) :]

        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)

    tensor_dims = [i for i, idx in enumerate(indices) if idx is not None]
    none_dims = [i for i, idx in enumerate(indices) if idx is None]
    remaining_dims = list(range(len(indices), inp.ndim))

    # Permutation that brings tensor-indexed dims to the front, as the kernel
    # always maps tensor_indices[i] to input dim i.
    perm = tensor_dims + none_dims + remaining_dims
    need_permute = perm != list(range(len(perm)))

    # target_shape (kernel order): [index_broadcast, none_dim_sizes, remaining_dims]
    for d in none_dims:
        target_shape.append(inp.shape[d])
    for d in remaining_dims:
        target_shape.append(inp.shape[d])

    tensor_indices = [indices[d] for d in tensor_dims]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [
        idx.to(torch.int32) if idx.dtype == torch.int64 else idx
        for idx in tensor_indices
    ]

    if values.device != inp.device:
        values = values.to(inp.device)
    if need_permute and values.ndim == len(perm):
        values = values.permute(perm)
    values = torch.broadcast_to(values, target_shape)

    out = inp.clone()
    if need_permute:
        _index_put_func(out.permute(perm), tensor_indices, values, accumulate)
    else:
        _index_put_func(out, tensor_indices, values, accumulate)
    return out


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS_ENFLAME INDEX_PUT_")

    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()
        target_shape = (K,) + inp.shape[len(indices) :]

        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)

    tensor_dims = [i for i, idx in enumerate(indices) if idx is not None]
    none_dims = [i for i, idx in enumerate(indices) if idx is None]
    remaining_dims = list(range(len(indices), inp.ndim))

    perm = tensor_dims + none_dims + remaining_dims
    need_permute = perm != list(range(len(perm)))

    for d in none_dims:
        target_shape.append(inp.shape[d])
    for d in remaining_dims:
        target_shape.append(inp.shape[d])

    tensor_indices = [indices[d] for d in tensor_dims]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [
        idx.to(torch.int32) if idx.dtype == torch.int64 else idx
        for idx in tensor_indices
    ]

    if values.device != inp.device:
        values = values.to(inp.device)
    if need_permute and values.ndim == len(perm):
        values = values.permute(perm)
    values = torch.broadcast_to(values, target_shape)

    if need_permute:
        _index_put_func(inp.permute(perm), tensor_indices, values, accumulate)
    else:
        _index_put_func(inp, tensor_indices, values, accumulate)
    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    if (
        len(indices) == 1
        and indices[0] is not None
        and indices[0].dtype in (torch.bool, torch.int8)
    ):
        mask = indices[0]
        if mask.device != inp.device:
            mask = mask.to(inp.device)
        if mask.dtype == torch.int8:
            mask = mask.bool()
        indices = [idx.to(inp.device) for idx in torch.where(mask.cpu())]
        K = indices[0].numel()
        target_shape = (K,) + inp.shape[len(indices) :]
        values = values.to(inp.device)
        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)
        else:
            values = values.broadcast_to(target_shape)
        tensor_indices = [
            idx.to(torch.int32) if idx.dtype == torch.int64 else idx for idx in indices
        ]
        _index_put_func(inp, tensor_indices, values, accumulate)
        return inp

    # step 1: index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    if len(indices) < inp.ndim:
        indices.extend([None] * (inp.ndim - len(indices)))

    if len(indices) > inp.ndim:
        raise IndexError("too many indices for tensor of dimension {}".format(inp.ndim))

    # Step 2: Broadcast tensor indices
    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

    # Step 3: Transpose
    is_contiguous = (tensor_pos[-1] - tensor_pos[0] + 1) == len(tensor_pos)
    starts_with_none = indices[0] is None
    need_transpose = not is_contiguous or starts_with_none

    if need_transpose:
        perm_order = tensor_pos + [i for i, x in enumerate(indices) if x is None]
        inp_view = inp.permute(perm_order)
        final_indices = [indices[i] for i in tensor_pos] + [None] * (
            len(indices) - len(tensor_pos)
        )
    else:
        inp_view = inp
        final_indices = indices

    # Step 4: Handle Values shape and broadcasting
    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]

    target_shape = broadcast_shape + slice_shape
    values = values.to(inp.device)
    if need_transpose and is_contiguous:
        num_before = tensor_pos[0]

        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)

        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        values = values.broadcast_to(target_shape)

    tensors = [
        idx.to(torch.int32) if idx.dtype == torch.int64 else idx for idx in tensors
    ]

    _index_put_func(inp_view, tensors, values, accumulate)

    return inp
