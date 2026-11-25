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

from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

# Maximum number of (merged) output dimensions supported by the strided
# kernel. Adjacent contiguous dims are merged on the host, so real layouts
# need far fewer lanes than their nominal rank.
_MAX_STRIDED_DIMS = 8
_FLAT_BLOCK = 4096
_STRIDED_BLOCK = 1024


@libentry()
@triton.jit
def _unbind_copy_flat_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # dim == 0 with a contiguous input: the slice-major output storage has the
    # exact same element order as the input storage, so a single flat copy is
    # enough. NEED_MASK is a constexpr so the exact-multiple case avoids the
    # (roughly 2x more expensive) masked-memory path.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < num_elements
        values = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, values, mask=mask)
    else:
        values = tl.load(input_ptr + offsets)
        tl.store(output_ptr + offsets, values)


@libentry()
@triton.jit
def unbind_copy_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    dim_prod_post,
    num_elements_per_slice,
    BLOCK_SIZE: tl.constexpr,
):
    # Contiguous input, dim != 0: each output slice s is a contiguous block of
    # the output storage; the input position of pre/post indices is strided.
    # 3D grid (pre, post chunk, slice) avoids any per-element div/mod.
    pre_idx = tl.program_id(0)
    post_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slice_idx = tl.program_id(2)
    mask = post_idx < dim_prod_post

    input_offset = (
        pre_idx * dim_size * dim_prod_post + slice_idx * dim_prod_post + post_idx
    )
    output_offset = (
        slice_idx * num_elements_per_slice + pre_idx * dim_prod_post + post_idx
    )
    values = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, values, mask=mask)


@libentry()
@triton.jit
def _unbind_copy_strided_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_elements_per_slice,
    dim_stride,
    dim_size_0,
    dim_size_1,
    dim_size_2,
    dim_size_3,
    dim_size_4,
    dim_size_5,
    dim_size_6,
    dim_size_7,
    dim_stride_0,
    dim_stride_1,
    dim_stride_2,
    dim_stride_3,
    dim_stride_4,
    dim_stride_5,
    dim_stride_6,
    dim_stride_7,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    # Non-contiguous input: the output storage is still a contiguous,
    # slice-major buffer, so the store stays affine (block DMA). The load has
    # to follow the real strides of the input view, including negative and
    # zero (broadcast) strides. Unused dim slots are padded with (1, 0) on the
    # host and removed by the constexpr guards below.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    slice_idx = offsets // num_elements_per_slice
    idx = offsets % num_elements_per_slice
    input_offset = slice_idx.to(tl.int64) * dim_stride

    if NUM_DIMS > 7:
        coord = idx % dim_size_7
        idx = idx // dim_size_7
        input_offset += coord.to(tl.int64) * dim_stride_7
    if NUM_DIMS > 6:
        coord = idx % dim_size_6
        idx = idx // dim_size_6
        input_offset += coord.to(tl.int64) * dim_stride_6
    if NUM_DIMS > 5:
        coord = idx % dim_size_5
        idx = idx // dim_size_5
        input_offset += coord.to(tl.int64) * dim_stride_5
    if NUM_DIMS > 4:
        coord = idx % dim_size_4
        idx = idx // dim_size_4
        input_offset += coord.to(tl.int64) * dim_stride_4
    if NUM_DIMS > 3:
        coord = idx % dim_size_3
        idx = idx // dim_size_3
        input_offset += coord.to(tl.int64) * dim_stride_3
    if NUM_DIMS > 2:
        coord = idx % dim_size_2
        idx = idx // dim_size_2
        input_offset += coord.to(tl.int64) * dim_stride_2
    if NUM_DIMS > 1:
        coord = idx % dim_size_1
        idx = idx // dim_size_1
        input_offset += coord.to(tl.int64) * dim_stride_1
    if NUM_DIMS > 0:
        coord = idx % dim_size_0
        input_offset += coord.to(tl.int64) * dim_stride_0

    values = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def _pick_block(n):
    b = triton.next_power_of_2(n)
    if b > 16384:
        b = 16384
    if b < 1024:
        b = 1024
    return b


def _merge_dims(sizes, strides):
    """Merge adjacent dims that are contiguous in the input view.

    A size-1 dim contributes nothing and is dropped. Dims ``k`` and ``k+1``
    (row-major order) merge when ``stride_k == size_{k+1} * stride_{k+1}``;
    the merged stride is then ``stride_{k+1}``.
    """
    merged_sizes = []
    merged_strides = []
    for size, stride in zip(sizes, strides):
        if size == 1:
            continue
        if merged_sizes and merged_strides[-1] == size * stride:
            merged_sizes[-1] *= size
            merged_strides[-1] = stride
        else:
            merged_sizes.append(size)
            merged_strides.append(stride)
    return merged_sizes, merged_strides


def _empty_outputs(output_shape, num_slices, dtype, device):
    return [
        torch.empty(output_shape, dtype=dtype, device=device) for _ in range(num_slices)
    ]


def unbind_copy(input: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
    if dim < 0:
        dim += input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    num_slices = input.shape[dim]
    if num_slices == 0:
        return []

    output_shape = list(input.shape)
    del output_shape[dim]
    num_elements_per_slice = 1
    for size in output_shape:
        num_elements_per_slice *= size
    if num_elements_per_slice == 0:
        return _empty_outputs(output_shape, num_slices, input.dtype, input.device)

    # torch.empty_strided instead of torch.empty: inside a gems context
    # torch.empty dispatches to the gems `empty` override (extra zeroing kernel
    # + per-call recompile in this environment); empty_strided is a pure native
    # allocation and is functionally identical for this contiguous storage.
    numel = num_slices * num_elements_per_slice
    output_storage = torch.empty_strided(
        (numel,), (1,), dtype=input.dtype, device=input.device
    )

    if input.is_contiguous():
        if dim == 0:
            # Contiguous input + dim == 0: flattened output storage has exactly
            # the same element order as the input, so the whole unbind is a
            # single flat copy (one kernel regardless of num_slices).
            block = _FLAT_BLOCK if numel <= 65536 else 8192
            num_warps = 4 if block == _FLAT_BLOCK else 8
            grid = (triton.cdiv(numel, block),)
            _unbind_copy_flat_kernel[grid](
                input,
                output_storage,
                numel,
                BLOCK_SIZE=block,
                NEED_MASK=numel % block != 0,
                num_warps=num_warps,
            )
        else:
            dim_prod_pre = 1
            for size in input.shape[:dim]:
                dim_prod_pre *= size
            dim_prod_post = 1
            for size in input.shape[dim + 1 :]:
                dim_prod_post *= size
            block_size = _pick_block(dim_prod_post)
            num_warps = 16 if block_size >= 16384 else (8 if block_size >= 2048 else 4)
            grid = (
                dim_prod_pre,
                triton.cdiv(dim_prod_post, block_size),
                num_slices,
            )
            unbind_copy_kernel[grid](
                input,
                output_storage,
                num_slices,
                dim_prod_post,
                num_elements_per_slice,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    else:
        # Arbitrary layout: address the input with its real strides (negative
        # and zero strides included) while keeping the output store affine.
        out_sizes = list(input.shape[:dim]) + list(input.shape[dim + 1 :])
        out_strides = list(input.stride()[:dim]) + list(input.stride()[dim + 1 :])
        out_sizes, out_strides = _merge_dims(out_sizes, out_strides)
        num_dims = len(out_sizes)
        if num_dims > _MAX_STRIDED_DIMS:
            raise NotImplementedError(
                f"unbind_copy: rank after merging contiguous dims is {num_dims}; "
                f"at most {_MAX_STRIDED_DIMS} are supported"
            )
        out_sizes += [1] * (_MAX_STRIDED_DIMS - num_dims)
        out_strides += [0] * (_MAX_STRIDED_DIMS - num_dims)
        grid = (triton.cdiv(numel, _STRIDED_BLOCK),)
        _unbind_copy_strided_kernel[grid](
            input,
            output_storage,
            numel,
            num_elements_per_slice,
            input.stride(dim),
            *out_sizes,
            *out_strides,
            BLOCK_SIZE=_STRIDED_BLOCK,
            NUM_DIMS=num_dims,
        )

    return [
        output_storage[
            slice_idx
            * num_elements_per_slice : (slice_idx + 1)
            * num_elements_per_slice
        ].reshape(output_shape)
        for slice_idx in range(num_slices)
    ]
