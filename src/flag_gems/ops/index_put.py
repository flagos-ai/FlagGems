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
import triton
import triton.language as tl

from flag_gems.ops.argsort import argsort
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


@triton.jit
def _offset(position, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full(position.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (position % SHAPE[axis]) * STRIDES[axis]
        position //= SHAPE[axis]
    return offset


@triton.jit
def _index_put_keys(
    Indices,
    Keys,
    COUNT,
    INDEX_SHAPE: tl.constexpr,
    INDEX_STRIDES: tl.constexpr,
    OUT_SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(COUNT, BLOCK), tl.num_programs(0)):
        position = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        key = tl.full((BLOCK,), 0, tl.int64)
        valid = position < COUNT
        for axis in tl.static_range(len(OUT_SHAPE)):
            cursor = position
            offset = tl.full((BLOCK,), 0, tl.int64)
            for dim in tl.static_range(len(INDEX_SHAPE) - 1, -1, -1):
                offset += (cursor % INDEX_SHAPE[dim]) * INDEX_STRIDES[axis][dim]
                cursor //= INDEX_SHAPE[dim]
            index = tl.load(Indices[axis] + offset, position < COUNT, 0).to(tl.int64)
            index = tl.where(index < 0, index + OUT_SHAPE[axis], index)
            valid &= (index >= 0) & (index < OUT_SHAPE[axis])
            key += index * OUT_STRIDES[axis]
        tl.store(Keys + position, tl.where(valid, key, -1), position < COUNT)


@triton.jit
def _index_put_ordered_kernel(
    Out,
    Values,
    Keys,
    Order,
    COUNT,
    SLICE_SIZE,
    SLICE_TILES,
    INDEX_SHAPE: tl.constexpr,
    SLICE_SHAPE: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    VALUE_INDEX_STRIDES: tl.constexpr,
    VALUE_SLICE_STRIDES: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(
        tl.program_id(0).to(tl.int64),
        COUNT.to(tl.int64) * SLICE_TILES,
        tl.num_programs(0),
    ):
        start = tile // SLICE_TILES
        position = tl.load(Order + start).to(tl.int64)
        key = tl.load(Keys + position)
        previous_position = tl.load(Order + start - 1, start > 0, 0).to(tl.int64)
        previous = tl.load(Keys + previous_position, start > 0, -1)
        if (key >= 0) & (key != previous):
            col = (tile % SLICE_TILES) * BLOCK + tl.arange(0, BLOCK)
            dest = key + _offset(col, SLICE_SHAPE, OUT_STRIDES)
            value_col = _offset(col, SLICE_SHAPE, VALUE_SLICE_STRIDES)
            valid = col < SLICE_SIZE
            acc = tl.load(Out + dest, valid, 0).to(ACC_DTYPE)
            row = start
            row_key = key
            while (row < COUNT) & (row_key == key):
                value_row = _offset(position, INDEX_SHAPE, VALUE_INDEX_STRIDES)
                acc += tl.load(Values + value_row + value_col, valid, 0).to(ACC_DTYPE)
                row += 1
                position = tl.load(Order + row, row < COUNT, 0).to(tl.int64)
                row_key = tl.load(Keys + position, row < COUNT, -1)
            tl.store(Out + dest, acc, valid)


def _index_put_ordered(inp, indices, values):
    indexed_dims = len(indices)
    index_shape = tuple(indices[0].shape)
    slice_shape = tuple(inp.shape[indexed_dims:])
    count = indices[0].numel()
    slice_size = values.numel() // count
    keys = torch.empty(count, dtype=torch.int64, device=inp.device)
    block = min(128, triton.next_power_of_2(slice_size))
    slice_tiles = triton.cdiv(slice_size, block)
    with torch_device_fn.device(inp.device):
        _index_put_keys[(min(triton.cdiv(count, 256), 65535),)](
            tuple(indices),
            keys,
            count,
            index_shape,
            tuple(index.stride() for index in indices),
            tuple(inp.shape[:indexed_dims]),
            inp.stride()[:indexed_dims],
            256,
        )
        # The merge argsort orders equal keys by their original positions.
        order = argsort(keys)
        _index_put_ordered_kernel[(min(count * slice_tiles, 65535),)](
            inp,
            values,
            keys,
            order,
            count,
            slice_size,
            slice_tiles,
            index_shape,
            slice_shape,
            inp.stride()[indexed_dims:],
            values.stride()[: len(index_shape)],
            values.stride()[len(index_shape) :],
            tl.float64 if inp.dtype == torch.float64 else tl.float32,
            block,
            num_warps=1,
        )
    return inp


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


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    # Keep every size-like argument a runtime value. triton specializes an
    # integer argument whose value is 1 into a compile-time constant, and these
    # sizes are used both in masks and in the div/mod chains that decompose the
    # offsets. When the index tensor holds a single element, `indices0_shape0`
    # (and M) therefore collapse to 1, `mask0 = offset0 < M` becomes a constant
    # mask, and the masked load's implicit constant `other` tensor makes
    # triton-ascend's TritonToLinalgIncubated pass abort on its way to linalg
    # (`llvm::cast<mlir::Attribute>` assertion on an OpFoldResult). None of these
    # arguments feed a block size or a pointer stride, so dropping their value
    # specialization costs nothing; the divisibility hints used for the strides
    # are separate and are kept.
    non_specialize = ["M", "N"]
    non_specialize += [f"input_shape{i}" for i in range(inp_rank)]
    non_specialize += [
        f"indices{i}_shape{j}" for i in range(indices_len) for j in range(index_rank)
    ]
    code.writeline(f"@triton.jit(do_not_specialize={non_specialize})")
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
            "BLOCK_SIZE0: tl.constexpr = 2,",
            "BLOCK_SIZE1: tl.constexpr = 2048,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("tiles0 = tl.cdiv(M, BLOCK_SIZE0)")
        code.writeline("tiles1 = tl.cdiv(N, BLOCK_SIZE1)")
        code.writeline("num_tiles = tiles0.to(tl.int64) * tiles1")
        code.writeline("first_tile = ext.program_id(axis=0).to(tl.int64)")
        code.writeline("for tile in range(first_tile, num_tiles, tl.num_programs(0)):")
        with code.indent():
            code.writeline("pid0 = tile // tiles1")
            code.writeline("pid1 = tile % tiles1")
            # Address arithmetic must stay in int64. With int32 indices,
            # `index * input_stride` can exceed 2**31 on large strided tensors and
            # wrap before it reaches the store, silently redirecting the write to a
            # wrapped address. Casting only the final pointer offset is too late,
            # so the program offsets are widened before the coordinates derived
            # from them. See Issue #6411.
            code.writeline(
                "offset0 = pid0.to(tl.int64) * BLOCK_SIZE0 + "
                "tl.arange(0, BLOCK_SIZE0).to(tl.int64)[:, None]"
            )
            if inp_rank == indices_len:
                code.writeline(
                    "offset1 = pid1.to(tl.int64) * 1 + "
                    "tl.arange(0, 1).to(tl.int64)[None, :]"
                )
            else:
                code.writeline(
                    "offset1 = pid1.to(tl.int64) * BLOCK_SIZE1 + "
                    "tl.arange(0, BLOCK_SIZE1).to(tl.int64)[None, :]"
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
                comp = [
                    f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)
                ]
                # The index tensor may be int32; widen the loaded index before it is
                # multiplied by input_stride so the product cannot wrap.
                code.writeline(
                    f"cur_index{i} = tl.load("
                    f"indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0"
                    ").to(tl.int64)"
                )
                code.writeline(
                    f"cur_index{i} = tl.where(cur_index{i} < 0, "
                    f"cur_index{i} + input_shape{i}, cur_index{i})"
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
                code.writeline(
                    "tl.store(input_ptr + input_offset, cur_value, mask=mask)"
                )

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
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.writeline("block_size1 = min(triton.next_power_of_2(N), 2048)")
        code.newline()
        code.writeline("def grid(meta):")
        with code.indent():
            code.writeline("tiles0 = triton.cdiv(M, meta['BLOCK_SIZE0'])")
            code.writeline("tiles1 = triton.cdiv(N, meta['BLOCK_SIZE1'])")
            code.writeline("return (min(tiles0 * tiles1, 65535),)")
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
            args += ["M,", "N,", "accumulate==True,", "BLOCK_SIZE1=block_size1,"]
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
    # Filter out None values to get actual tensor indices
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
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
        inp, tensor_indices, values, accumulate = args
        if values.numel() == 0:
            return inp
        if (
            accumulate
            and device.vendor_name == "hygon"
            and (
                inp.dtype in (torch.float16, torch.bfloat16)
                or (
                    inp.is_floating_point()
                    and torch.are_deterministic_algorithms_enabled()  # allow-torch-compute: reads a runtime flag
                )
            )
        ):
            return _index_put_ordered(inp, tensor_indices, values)
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
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

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices, _ = args[0], args[1], args[2]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_put_func = IndexPutFunction()


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX_PUT")

    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX_PUT_")

    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")
    if values.dtype != inp.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes to match, "
            f"got {values.dtype} and {inp.dtype}"
        )

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]
    # step 1: index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.uint8):
            axis = len(processed_indices)
            if idx.ndim and idx.shape != inp.shape[axis : axis + idx.ndim]:
                raise IndexError(
                    "The shape of the mask must match the indexed dimensions"
                )
            # Expand bool masks into explicit integer indices
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif idx.dtype in (torch.int32, torch.int64):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    # Pad missing None indices to match input dimension
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

        # 1. Broadcast to PyTorch natural shape
        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)

        # 2. Permute to Kernel expectation
        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        # direct broadcast
        values = values.broadcast_to(target_shape)

    _index_put_func(inp_view, tensors, values, accumulate)

    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS _INDEX_PUT_IMPL_")
    return index_put_(inp, indices, values, accumulate)
