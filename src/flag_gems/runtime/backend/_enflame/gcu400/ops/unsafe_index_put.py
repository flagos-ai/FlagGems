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
from typing import Any, Callable, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)

# Target CTA count for the fully indexed case (inp_rank == indices_len), where
# each CTA writes BLOCK_SIZE0 consecutive scattered elements and the launch is
# bound by the number of CTAs rather than by memory traffic.
_FULLY_INDEXED_TARGET_CTAS = 256

# Default 2D tile for the sliced case (rows x slice elements per CTA).
_DEFAULT_BLOCK_SIZE0 = 2
_DEFAULT_BLOCK_SIZE1 = 2048

# A sliced launch is CTA-bound once it needs more than this many CTAs: few rows
# over a long slice (e.g. 2x128 indices into a 512^3 tensor) then costs one tiny
# CTA per 2048 slice elements. Above the threshold the tile switches to many
# rows over a shorter slice block, which cuts the CTA count 4x.
_SLICED_MAX_CTAS = 2048
_SLICED_WIDE_TILE = 16384
_SLICED_MAX_ROWS = 64


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
            "SLICE_UNIFORM_VALUES: tl.constexpr = False,",
            "BLOCK_SIZE0: tl.constexpr = 2,",
            "BLOCK_SIZE1: tl.constexpr = 2048,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
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
        uniform_comp = list(comp)
        comp += [
            f"input_idx{indices_len + i} * values_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.newline()
        if inp_rank > indices_len:
            # A broadcast value (torch.index_put with a scalar or a per-index-row
            # value) has stride 0 on every slice axis. Loading such a tile as a
            # full 2D gather turns into one address computation and one scalar
            # load per lane; hoisting the load to the row axis keeps it at
            # BLOCK_SIZE0 lanes and lets the tile broadcast instead.
            code.writeline("if SLICE_UNIFORM_VALUES:")
            with code.indent():
                code.writeline(f"values_offset = {' + '.join(uniform_comp)}")
                code.writeline(
                    "cur_value = tl.load("
                    "values_ptr + values_offset, mask=mask0 & index_mask, other=0)"
                )
            code.writeline("else:")
            with code.indent():
                code.writeline(f"values_offset = {' + '.join(comp)}")
                code.writeline(
                    "cur_value = tl.load(values_ptr + values_offset, mask=mask)"
                )
        else:
            code.writeline(f"values_offset = {' + '.join(comp)}")
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
        # Block sizing. Every CTA owns BLOCK_SIZE0 rows (one gathered row per
        # index entry) times BLOCK_SIZE1 slice elements.
        if inp_rank == indices_len:
            # Fully indexed input: the slice axis is dead, so the 2D tile is
            # BLOCK_SIZE0 x 1 and the CTA count dominates the cost. The kernel
            # default BLOCK_SIZE0=2 turns a (2**28,) scatter of 2**16 indices
            # into 32768 two-element CTAs, which is ~10x slower than the 1D
            # copy it replaces. Pick the row block from a target CTA count
            # instead, clamped to the 8192-lane tile limit.
            code.writeline(
                "rows_per_cta = max(1, triton.cdiv(M, %d))" % _FULLY_INDEXED_TARGET_CTAS
            )
            code.writeline(
                "BLOCK_SIZE0 = min(8192, max(2, 1 << (rows_per_cta - 1).bit_length()))"
            )
            code.writeline("BLOCK_SIZE1 = 1")
        else:
            # Sliced input. grid.y <= 255, so BLOCK_SIZE1 must cover N/255 at
            # least; grid.x <= 65535 is handled by the need_b0 guard below.
            code.writeline("BLOCK_SIZE0 = %d" % _DEFAULT_BLOCK_SIZE0)
            code.writeline("need_b1 = max(1, triton.cdiv(N, 255))")
            code.writeline(
                "BLOCK_SIZE1 = max(%d, 1 << (need_b1 - 1).bit_length())"
                % _DEFAULT_BLOCK_SIZE1
            )
            # Few rows over a long slice needs one tiny CTA per BLOCK_SIZE1
            # slice elements; use the extra rows per CTA instead (see
            # _SLICED_MAX_CTAS).
            code.writeline(
                "if triton.cdiv(M, BLOCK_SIZE0) * triton.cdiv(N, BLOCK_SIZE1) > %d:"
                % _SLICED_MAX_CTAS
            )
            with code.indent():
                code.writeline(
                    "BLOCK_SIZE0 = min(%d, 1 << (max(M, 1) - 1).bit_length(),"
                    " max(1, %d // BLOCK_SIZE1))"
                    % (_SLICED_MAX_ROWS, _SLICED_WIDE_TILE)
                )
            code.writeline("need_b0 = max(2, triton.cdiv(M, 65535))")
            code.writeline("if need_b0 > BLOCK_SIZE0:")
            with code.indent():
                code.writeline("BLOCK_SIZE0 = 1 << (need_b0 - 1).bit_length()")
                # Keep the 2D tile at most 8192 lanes; the row block grows only
                # on the large-M path, so shrink the slice block to compensate.
                code.writeline(
                    "BLOCK_SIZE1 = max(1, min(BLOCK_SIZE1, 8192 // BLOCK_SIZE0))"
                )
        code.writeline(
            "grid = (triton.cdiv(M, BLOCK_SIZE0), triton.cdiv(N, BLOCK_SIZE1))"
        )
        if inp_rank > indices_len:
            # The value tensor only has to be gathered per slice element when at
            # least one slice stride is non-zero; a broadcasted value keeps the
            # cheap row-only load.
            slice_strides = [
                f"values_stride[{index_rank + i}]"
                for i in range(inp_rank - indices_len)
            ]
            code.writeline(
                "slice_uniform_values = all(s == 0 for s in ("
                + ", ".join(slice_strides)
                + ",))"
            )
        else:
            code.writeline("slice_uniform_values = False")
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
            args += [
                "M,",
                "N,",
                "accumulate==True,",
                "SLICE_UNIFORM_VALUES=slice_uniform_values,",
                "BLOCK_SIZE0=BLOCK_SIZE0,",
                "BLOCK_SIZE1=BLOCK_SIZE1,",
            ]
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


class UnsafeIndexPutFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, values, accumulate = args
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_unsafe_index_put_wrapper",
                "_unsafe_index_put_jit_function",
                code,
            )
            file_name = f"unsafe_index_put_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_unsafe_index_put_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_unsafe_index_put_wrapper")
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


_unsafe_index_put_func = UnsafeIndexPutFunction()


def unsafe_index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS_ENFLAME UNSAFE_INDEX_PUT")

    # The accumulation path (accumulate=True) uses tl.atomic_add to accumulate in place in the input dtype:
    # native atomic adds for fp16/bf16 round to low precision after every add; with repeated indices (especially when a
    # single
    # add makes the magnitude far larger than the final result and cancellation occurs), the accumulated error exceeds
    # the test tolerance.
    # Therefore, the accumulation path for low-precision inputs first promotes everything to fp32 for accumulation, then
    # writes back to the original dtype once at the end.
    if accumulate and inp.dtype in (torch.float16, torch.bfloat16):
        fp32_accumulator = inp.float()
        unsafe_index_put_impl(fp32_accumulator, indices, values.float(), accumulate)
        return fp32_accumulator.to(inp.dtype)

    out = inp.clone()
    return unsafe_index_put_impl(out, indices, values, accumulate)


def unsafe_index_put_impl(inp, indices, values, accumulate=False):
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
    # step 1: index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            # Expand bool masks into explicit integer indices
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
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

    _unsafe_index_put_func(inp_view, tensors, values, accumulate)

    return inp
