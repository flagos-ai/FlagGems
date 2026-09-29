# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _concat_copy_flat_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    OUTPUT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    value = tl.load(
        input_ptr + offsets,
        mask=mask,
    )

    tl.store(
        output_ptr + OUTPUT_OFFSET + offsets,
        value,
        mask=mask,
    )


@libentry()
@triton.jit
def _concat_copy_strided_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    INNER_SIZE: tl.constexpr,
    INPUT_CAT_SIZE: tl.constexpr,
    OUTPUT_CAT_SIZE: tl.constexpr,
    CAT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    inner_idx = offsets % INNER_SIZE

    tmp = offsets // INNER_SIZE

    cat_idx = tmp % INPUT_CAT_SIZE
    outer_idx = tmp // INPUT_CAT_SIZE

    output_offsets = (
        outer_idx * OUTPUT_CAT_SIZE + CAT_OFFSET + cat_idx
    ) * INNER_SIZE + inner_idx

    value = tl.load(
        input_ptr + offsets,
        mask=mask,
    )

    tl.store(
        output_ptr + output_offsets,
        value,
        mask=mask,
    )


def _prepare_inputs(A, dim):
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    tensors = list(A)

    device = tensors[0].device
    dtype = tensors[0].dtype

    # PyTorch cat allows 1-D empty tensors with shape (0,)
    tensors = [t for t in tensors if t.shape != torch.Size([0])]

    if len(tensors) == 0:
        return tensors, 0, device, dtype, [0]

    ndim = tensors[0].ndim

    if ndim == 0:
        raise RuntimeError("zero-dimensional tensor cannot be concatenated")

    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range "
            f"(expected to be in range of "
            f"[-{ndim}, {ndim - 1}], but got {dim})"
        )

    dim %= ndim

    base_shape = list(tensors[0].shape)

    for idx, t in enumerate(tensors):
        shape = list(t.shape)

        if t.device != device:
            raise RuntimeError("Expected all tensors to be on the same device")

        if t.dtype != dtype:
            raise RuntimeError("Expected all tensors to have the same dtype")

        if len(shape) != ndim:
            raise RuntimeError("Tensors must have same number of dimensions")

        for axis in range(ndim):
            if axis == dim:
                continue

            if shape[axis] != base_shape[axis]:
                raise RuntimeError(
                    "Sizes of tensors must match except "
                    f"in dimension {dim}. "
                    f"Expected size {base_shape[axis]} "
                    f"but got size {shape[axis]} "
                    f"for tensor number {idx}"
                )

    out_shape = base_shape.copy()

    out_shape[dim] = sum(t.shape[dim] for t in tensors)

    return tensors, dim, device, dtype, out_shape


def concat(
    A: Union[
        Tuple[torch.Tensor, ...],
        List[torch.Tensor],
    ],
    dim: int = 0,
) -> torch.Tensor:
    A, dim, device, dtype, out_shape = _prepare_inputs(
        A,
        dim,
    )

    if len(A) == 0:
        return torch.empty(
            out_shape,
            device=device,
            dtype=dtype,
        )

    out = torch.empty(
        out_shape,
        device=device,
        dtype=dtype,
    )

    if out.numel() == 0:
        return out

    BLOCK_SIZE = 256

    output_cat_size = out.shape[dim]

    inner_size = math.prod(out.shape[dim + 1 :])

    cat_offset = 0

    with torch_device_fn.device(out.device):
        for a in A:
            input_cat_size = a.shape[dim]

            if a.numel() == 0:
                cat_offset += input_cat_size
                continue

            # Keep Triton indexing simple.
            # Main concat copy is still performed by Triton.
            if not a.is_contiguous():
                a = a.contiguous()

            n_elements = a.numel()

            grid = (
                triton.cdiv(
                    n_elements,
                    BLOCK_SIZE,
                ),
            )

            # dim=0 maps the entire tensor to one contiguous
            # output interval.
            if dim == 0:
                output_offset = cat_offset * inner_size

                _concat_copy_flat_kernel[grid](
                    a,
                    out,
                    n_elements,
                    OUTPUT_OFFSET=output_offset,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

            else:
                _concat_copy_strided_kernel[grid](
                    a,
                    out,
                    n_elements,
                    INNER_SIZE=inner_size,
                    INPUT_CAT_SIZE=input_cat_size,
                    OUTPUT_CAT_SIZE=output_cat_size,
                    CAT_OFFSET=cat_offset,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

            cat_offset += input_cat_size

    return out
