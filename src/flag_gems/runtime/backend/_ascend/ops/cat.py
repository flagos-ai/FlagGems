import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _cat_copy_flat_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    OUTPUT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    values = tl.load(
        input_ptr + offsets,
        mask=mask,
    )

    tl.store(
        output_ptr + OUTPUT_OFFSET + offsets,
        values,
        mask=mask,
    )


@libentry()
@triton.jit
def _cat_copy_strided_kernel(
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

    values = tl.load(
        input_ptr + offsets,
        mask=mask,
    )

    tl.store(
        output_ptr + output_offsets,
        values,
        mask=mask,
    )


def _prepare_cat_inputs(A, dim):
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    device = A[0].device
    dtype = A[0].dtype

    A = list(A)

    for i in range(len(A) - 1, -1, -1):
        if A[i].shape == torch.Size([0]):
            A.pop(i)

    if len(A) == 0:
        return A, dim, device, dtype, [0]

    ndim = A[0].ndim

    if not (-ndim <= dim < ndim):
        raise IndexError(
            f"Dimension out of range " f"(expected [-{ndim}, {ndim - 1}], got {dim})"
        )

    dim %= ndim

    shapes = [list(t.shape) for t in A]
    base = shapes[0]

    for tensor_idx, shape in enumerate(shapes):
        if len(shape) != len(base):
            raise RuntimeError("Tensors must have same number of dimensions")

        for axis, (expected, actual) in enumerate(zip(base, shape)):
            if axis == dim:
                continue

            if expected != actual:
                raise RuntimeError(
                    "Sizes of tensors must match except "
                    f"in dimension {dim}: expected {expected}, "
                    f"got {actual} for tensor {tensor_idx}"
                )

    out_shape = list(base)
    out_shape[dim] = sum(s[dim] for s in shapes)

    return A, dim, device, dtype, out_shape


def _cat_fill_triton(out, A, dim):
    if len(A) == 0 or out.numel() == 0:
        return

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

            if not a.is_contiguous():
                a = a.contiguous()

            n_elements = a.numel()

            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            if dim == 0:
                output_offset = cat_offset * inner_size

                _cat_copy_flat_kernel[grid](
                    a,
                    out,
                    n_elements,
                    OUTPUT_OFFSET=output_offset,
                    BLOCK_SIZE=BLOCK_SIZE,
                )

            else:
                _cat_copy_strided_kernel[grid](
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


def cat(
    A: Union[
        Tuple[torch.Tensor, ...],
        List[torch.Tensor],
    ],
    dim: int = 0,
):
    A, dim, device, dtype, out_shape = _prepare_cat_inputs(A, dim)

    if len(A) == 0:
        return torch.empty(
            out_shape,
            device=device,
            dtype=dtype,
        )

    if len(A) == 1:
        return A[0]

    out = torch.empty(
        out_shape,
        dtype=dtype,
        device=device,
    )

    _cat_fill_triton(out, A, dim)

    return out


def cat_out(
    A: Union[
        Tuple[torch.Tensor, ...],
        List[torch.Tensor],
    ],
    dim: int = 0,
    *,
    out: torch.Tensor,
):
    A, dim, device, dtype, out_shape = _prepare_cat_inputs(A, dim)

    out.resize_(out_shape)

    if len(A) == 0:
        return out

    if len(A) == 1:
        out.copy_(A[0])
        return out

    if out.is_contiguous():
        _cat_fill_triton(out, A, dim)
    else:
        tmp = torch.empty(
            out_shape,
            dtype=out.dtype,
            device=out.device,
        )

        _cat_fill_triton(tmp, A, dim)
        out.copy_(tmp)

    return out
