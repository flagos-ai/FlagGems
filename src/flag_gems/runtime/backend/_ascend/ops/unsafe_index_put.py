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

"""Ascend-specialized ``unsafe_index_put`` (aten._unsafe_index_put).

The generic triton kernel scatters ``tl.atomic_add`` on global memory for the
accumulate path. The TritonAscend backend only executes those atomics correctly
while the grid fits in a single wave (total number of programs <= the number of
vector cores, 40 on 910B): with a larger grid the extra waves produce lost and
misdirected atomic adds, corrupting the result.

This variant keeps the accumulate path in triton by capping the grid so the
launch fits in one wave and serializing the remaining work with inner loops:
each program processes ``iter0`` consecutive BLOCK_SIZE0 chunks along the index
dimension and (when the input has trailing sliced dimensions) ``iter1`` chunks
along the trailing dimension. Atomics from a single wave are exact on Ascend,
so the capped kernel is correct regardless of the original grid size.

For fp16/bf16 inputs the accumulate path additionally runs on an fp32 working
buffer and casts back: fp16/bf16 in-place atomic adds round every addition and
can drift beyond tolerance from the fp64 reference, while fp32 accumulation
error stays far below one fp16/bf16 ULP. The non-accumulate path (plain masked
stores, no atomics) reuses the generic implementation unchanged.
"""

import functools
import importlib
import logging
from typing import Any, Tuple

import torch

from flag_gems.ops.unsafe_index_put import UnsafeIndexPutFunction, generate_imports
from flag_gems.ops.unsafe_index_put import unsafe_index_put as _generic_unsafe_index_put
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _vector_core_num(device_index: int) -> int:
    """Number of vector cores on the device, i.e. programs per wave."""
    try:
        import triton.runtime.driver as driver

        props = driver.active.utils.get_device_properties(device_index)
        return int(props["num_vectorcore"])
    except Exception:
        return CORE_NUM


def _emit_index_put_body(
    code: IndentedBuffer, inp_rank: int, indices_len: int, index_rank: int
):
    """Emit index decomposition, loads and the atomic/store at current indent."""
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
        code.writeline("tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask)")
    code.writeline("else:")
    with code.indent():
        code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")


def generate_index_put_kernel_ascend(
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
            "BLOCK_SIZE0: tl.constexpr = 2,",
            "BLOCK_SIZE1: tl.constexpr = 2048,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
        code.writeline("total0 = tl.cdiv(M, BLOCK_SIZE0)")
        code.writeline("iter0 = tl.cdiv(total0, ext.num_programs(axis=0).to(tl.int32))")
        code.writeline("for i0 in range(0, iter0):")
        with code.indent():
            code.writeline(
                "offset0 = (pid0 * iter0 + i0) * BLOCK_SIZE0"
                " + tl.arange(0, BLOCK_SIZE0)[:, None]"
            )
            if inp_rank == indices_len:
                code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
                _emit_index_put_body(code, inp_rank, indices_len, index_rank)
            else:
                code.writeline("total1 = tl.cdiv(N, BLOCK_SIZE1)")
                code.writeline(
                    "iter1 = tl.cdiv(total1, ext.num_programs(axis=1).to(tl.int32))"
                )
                code.writeline("for i1 in range(0, iter1):")
                with code.indent():
                    code.writeline(
                        "offset1 = (pid1 * iter1 + i1) * BLOCK_SIZE1"
                        " + tl.arange(0, BLOCK_SIZE1)[None, :]"
                    )
                    _emit_index_put_body(code, inp_rank, indices_len, index_rank)

    code.newline()
    code.newline()
    return code


def generate_index_put_wrapper_ascend(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(
        f"def {wrapper_name}(input, indices, values, accumulate, max_programs):"
    )
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
        # Cap the total grid to max_programs so the launch fits in a single
        # wave; the kernel serializes the remaining chunks in inner loops.
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("min(triton.cdiv(M, meta['BLOCK_SIZE0']), max_programs),")
            code.writeline("min(")
            with code.indent():
                code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']),")
                code.writeline(
                    "max(1, max_programs // min("
                    "triton.cdiv(M, meta['BLOCK_SIZE0']), max_programs)"
                    "),"
                )
            code.writeline("),")
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


def generate_code_ascend(
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
    generate_index_put_kernel_ascend(
        inp_rank, indices_len, index_rank, kernel_name, code
    )
    generate_index_put_wrapper_ascend(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class AscendUnsafeIndexPutFunction(UnsafeIndexPutFunction):
    def __call__(self, *args, **kwargs):
        inp, tensor_indices, values, accumulate = args
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code_ascend(
                full_args,
                "_unsafe_index_put_wrapper",
                "_unsafe_index_put_jit_function",
                code,
            )
            file_name = f"unsafe_index_put_ascend_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_unsafe_index_put_ascend_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_unsafe_index_put_wrapper")
            self.overloads[key] = overload

        return overload(*args, _vector_core_num(inp.device.index or 0))


_ascend_unsafe_index_put_func = AscendUnsafeIndexPutFunction()


def unsafe_index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS_ASCEND UNSAFE_INDEX_PUT")

    if not accumulate:
        # Plain masked stores only: the generic triton kernel is exact on
        # Ascend for any grid size, keep it for the non-atomic path.
        return _generic_unsafe_index_put(inp, indices, values, accumulate)

    out = inp.clone()
    if inp.dtype in (torch.float16, torch.bfloat16):
        # fp16/bf16 in-place atomic adds round each addition, which can drift
        # beyond tolerance from the fp64 reference. Accumulate in an fp32
        # buffer instead and cast back: fp32 accumulation error is far below
        # one fp16/bf16 ULP.
        work = out.to(torch.float32)
        work = unsafe_index_put_impl(
            work, indices, values.to(torch.float32), accumulate
        )
        out.copy_(work)
        return out
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
    for pos, idx in enumerate(indices):
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.uint8):
            # aten accepts bool and (deprecated) uint8 indices as masks; every
            # mask dim must match the input dim at the mask's position in the
            # index list. Expand the mask into explicit integer indices.
            bad_dim = next(
                (
                    i
                    for i in range(idx.ndim)
                    if pos + i >= inp.ndim or idx.shape[i] != inp.shape[pos + i]
                ),
                None,
            )
            if bad_dim is not None:
                raise IndexError(
                    f"The shape of the mask {list(idx.shape)} at index {bad_dim} does not match "
                    f"the shape of the indexed tensor {list(inp.shape)} at index {pos + bad_dim}"
                )
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx) and idx.dtype in (torch.int32, torch.int64):
            processed_indices.append(idx)
        else:
            raise IndexError(
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

    _ascend_unsafe_index_put_func(inp_view, tensors, values, accumulate)

    return inp
