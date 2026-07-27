# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit
def split_with_sizes_copy_kernel(
    out_ptr,
    inp_ptr,
    split_start,
    split_size,
    dim_size,
    dim_prod_post,
    BLOCK_SIZE: tl.constexpr,
):
    pre_idx = tl.program_id(0)
    post_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    split_width = split_size * dim_prod_post
    mask = post_idx < split_width

    out_offset = pre_idx * split_width + post_idx
    inp_offset = (
        pre_idx * dim_size * dim_prod_post
        + split_start * dim_prod_post
        + post_idx
    )
    values = tl.load(inp_ptr + inp_offset, mask=mask)
    tl.store(out_ptr + out_offset, values, mask=mask)


def _normalize_split_sizes(split_sizes):
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = split_sizes.tolist()
    return [int(size) for size in split_sizes]


def _normalize_dim(input, dim):
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    return dim % input.ndim


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def split_with_sizes_copy(input, split_sizes, dim=0):
    dim = _normalize_dim(input, dim)
    split_sizes = _normalize_split_sizes(split_sizes)
    assert all(size >= 0 for size in split_sizes), "Invalid split_sizes"
    assert sum(split_sizes) == input.shape[dim], "Invalid split_sizes"

    source = input if input.is_contiguous() else input.contiguous()
    dim_size = input.shape[dim]
    dim_prod_pre = _product(input.shape[:dim])
    dim_prod_post = _product(input.shape[dim + 1 :])
    block_size = 1024
    outputs = []
    split_start = 0

    for split_size in split_sizes:
        output_shape = list(input.shape)
        output_shape[dim] = split_size
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        if output.numel() != 0:
            grid = (
                dim_prod_pre,
                triton.cdiv(split_size * dim_prod_post, block_size),
            )
            split_with_sizes_copy_kernel[grid](
                output,
                source,
                split_start,
                split_size,
                dim_size,
                dim_prod_post,
                BLOCK_SIZE=block_size,
            )
        outputs.append(output)
        split_start += split_size

    return tuple(outputs)
