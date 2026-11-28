import math
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_ASCEND_MAX_CORE_DIM = 65535


@libentry()
@triton.jit
def _cat_copy_flat_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    OUTPUT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_SIZE)
    for block in range(
        tl.program_id(0), tl.cdiv(n_elements, BLOCK_SIZE), tl.num_programs(0)
    ):
        offsets = block * BLOCK_SIZE + lanes
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


def _prepare_cat_inputs(A, dim):
    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    device = A[0].device

    A = list(A)
    dtype = A[0].dtype
    for tensor in A[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)

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

    A = [tensor.to(dtype) if tensor.dtype != dtype else tensor for tensor in A]

    return A, dim, device, dtype, out_shape


def _cat_fill_triton(out, A, dim):
    if len(A) == 0 or out.numel() == 0:
        return

    element_size = out.element_size()
    # Wide tiles amortise the per-block launch cost, but BiShengIR double
    # buffers the loaded tile: keep 2 * BLOCK_SIZE * element_size inside the
    # 192 KB unified buffer (32K fp16 works, 32K fp32 overflows).
    if element_size <= 2:
        flat_block = 32768
    elif element_size <= 4:
        flat_block = 16384
    else:
        flat_block = 8192

    inner_size = math.prod(out.shape[dim + 1 :])

    if inner_size == 0:
        return

    cat_offset = 0

    with torch_device_fn.device(out.device):
        for a in A:
            input_cat_size = a.shape[dim]

            if a.numel() == 0:
                cat_offset += input_cat_size
                continue

            if dim == 0 and a.is_contiguous():
                n_elements = a.numel()
                # CANN limits the flattened launch grid (coreDim) to 65535.
                grid = (min(triton.cdiv(n_elements, flat_block), _ASCEND_MAX_CORE_DIM),)
                output_offset = cat_offset * inner_size

                _cat_copy_flat_kernel[grid](
                    a,
                    out,
                    n_elements,
                    OUTPUT_OFFSET=output_offset,
                    BLOCK_SIZE=flat_block,
                )

            else:
                # dim > 0 (and non-contiguous input for dim == 0): a Triton
                # scatter of inner_size-sized rows is integer-bound and orders
                # of magnitude slower than the vendor layout copy, so write
                # straight into the destination slice.
                out.narrow(dim, cat_offset, input_cat_size).copy_(a)

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

    if not torch.can_cast(dtype, out.dtype):
        raise RuntimeError(
            f"torch.cat(): input types can't be cast to the desired output type "
            f"{out.dtype}"
        )

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
