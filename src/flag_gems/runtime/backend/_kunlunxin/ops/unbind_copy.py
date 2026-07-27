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

from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def unbind_copy_dim0_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def unbind_copy_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    dim_prod_post,
    num_elements_per_slice,
    BLOCK_SIZE: tl.constexpr,
):
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
        return [
            torch.empty(output_shape, dtype=input.dtype, device=input.device)
            for _ in range(num_slices)
        ]

    dim_prod_pre = 1
    for size in input.shape[:dim]:
        dim_prod_pre *= size
    dim_prod_post = 1
    for size in input.shape[dim + 1 :]:
        dim_prod_post *= size

    output_storage = torch.empty(
        (num_slices * num_elements_per_slice,), dtype=input.dtype, device=input.device
    )
    source = input if input.is_contiguous() else input.contiguous()
    block_size = 1024
    if dim == 0:
        unbind_copy_dim0_kernel[(triton.cdiv(source.numel(), block_size),)](
            source,
            output_storage,
            source.numel(),
            BLOCK_SIZE=block_size,
        )
    else:
        grid = (dim_prod_pre, triton.cdiv(dim_prod_post, block_size), num_slices)
        unbind_copy_kernel[grid](
            source,
            output_storage,
            num_slices,
            dim_prod_post,
            num_elements_per_slice,
            BLOCK_SIZE=block_size,
        )

    return [
        output_storage[
            slice_idx * num_elements_per_slice : (slice_idx + 1)
            * num_elements_per_slice
        ].reshape(output_shape)
        for slice_idx in range(num_slices)
    ]
