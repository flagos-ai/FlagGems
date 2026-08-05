import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


@libentry()
@triton.jit
def diagonal_lastdim_optimized_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    diag_len: tl.constexpr,
    batch_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < (batch_numel * diag_len)

    batch_idx = idx // diag_len
    diag_idx = idx % diag_len

    input_offsets = batch_idx * rows * cols + diag_idx * (cols + 1)
    val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + idx, val, mask=mask)


@libentry()
@triton.jit
def diagonal_strided_kernel(
    input_tensor,
    output_tensor,
    out_size0,
    out_size1,
    out_size2,
    out_size3,
    out_size4,
    in_stride0,
    in_stride1,
    in_stride2,
    in_stride3,
    in_stride4,
    STORAGE_OFFSET: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel

    c4 = idx % out_size4
    rem4 = idx // out_size4
    c3 = rem4 % out_size3
    rem3 = rem4 // out_size3
    c2 = rem3 % out_size2
    rem2 = rem3 // out_size2
    c1 = rem2 % out_size1
    c0 = rem2 // out_size1

    in_offset = (
        c0 * in_stride0
        + c1 * in_stride1
        + c2 * in_stride2
        + c3 * in_stride3
        + c4 * in_stride4
    )
    ptr = input_tensor + STORAGE_OFFSET + in_offset
    val = tl.load(ptr, mask=mask, other=0.0)
    tl.store(output_tensor + idx, val, mask=mask)


def linalg_diagonal(
    A: torch.Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> torch.Tensor:
    logger.debug("GEMS Triton diagonal")

    if A.dim() < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
    ndim = A.dim()
    dim1 = _normalize_dim(dim1, ndim)
    dim2 = _normalize_dim(dim2, ndim)
    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same")
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1

    size1 = A.shape[dim1]
    size2 = A.shape[dim2]
    if offset >= 0:
        diag_len = max(0, min(size1, size2 - offset))
    else:
        diag_len = max(0, min(size1 + offset, size2))

    out_shape = list(A.shape)
    for d in sorted([dim1, dim2], reverse=True):
        out_shape.pop(d)
    out_shape.insert(max(dim1, dim2), diag_len)

    if diag_len == 0:
        return torch.empty(tuple(out_shape), dtype=A.dtype, device=A.device)

    if offset == 0 and dim1 == ndim - 2 and dim2 == ndim - 1 and A.is_contiguous():
        batch_shape = A.shape[:-2]
        batch_numel = 1
        for s in batch_shape:
            batch_numel *= s
        rows = A.shape[-2]
        cols = A.shape[-1]
        total_elems = batch_numel * diag_len

        BLOCK_SIZE = 256 if diag_len <= 256 else 1024
        grid = (triton.cdiv(total_elems, BLOCK_SIZE),)

        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        with torch_device_fn.device(out.device):
            diagonal_lastdim_optimized_kernel[grid](
                A,
                out,
                rows=rows,
                cols=cols,
                diag_len=diag_len,
                batch_numel=batch_numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return out

    in_strides = list(A.stride())
    inter_strides = in_strides.copy()
    for d in sorted([dim1, dim2], reverse=True):
        inter_strides.pop(d)
    diag_stride = in_strides[dim1] + in_strides[dim2]
    inter_strides.insert(max(dim1, dim2), diag_stride)

    storage_off = A.storage_offset()
    if offset >= 0:
        storage_off += offset * in_strides[dim2]
    else:
        storage_off += (-offset) * in_strides[dim1]

    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
    out_sizes = list(out.shape)
    pad = 5 - len(out_sizes)
    out_sizes_padded = [1] * pad + out_sizes
    in_strides_padded = [0] * pad + inter_strides

    numel = out.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(numel))
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    with torch_device_fn.device(out.device):
        diagonal_strided_kernel[grid](
            A,
            out,
            out_sizes_padded[0],
            out_sizes_padded[1],
            out_sizes_padded[2],
            out_sizes_padded[3],
            out_sizes_padded[4],
            in_strides_padded[0],
            in_strides_padded[1],
            in_strides_padded[2],
            in_strides_padded[3],
            in_strides_padded[4],
            STORAGE_OFFSET=storage_off,
            numel=numel,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out
