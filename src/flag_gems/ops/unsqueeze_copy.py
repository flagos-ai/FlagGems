import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _unsqueeze_copy_kernel(
    src_ptr,
    dst_ptr,
    sizes_ptr,
    src_strides_ptr,
    dst_strides_ptr,
    n_elements,
    NDIM: tl.constexpr,
    INSERT_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    offsets = offsets.to(tl.int64)

    src_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    remain = offsets

    for d in range(NDIM - 1, -1, -1):
        size = tl.load(sizes_ptr + d)

        index = remain % size
        remain = remain // size

        src_stride = tl.load(src_strides_ptr + d)
        src_offset += index * src_stride

        if d < INSERT_DIM:
            dst_stride = tl.load(dst_strides_ptr + d)
        else:
            dst_stride = tl.load(dst_strides_ptr + d + 1)

        dst_offset += index * dst_stride

    values = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, values, mask=mask)


def _launch_unsqueeze_copy(
    src: torch.Tensor,
    dim: int,
    out: torch.Tensor,
):
    n_elements = src.numel()

    if n_elements == 0:
        return out

    sizes = torch.tensor(
        list(src.shape),
        dtype=torch.int64,
        device=src.device,
    )

    src_strides = torch.tensor(
        list(src.stride()),
        dtype=torch.int64,
        device=src.device,
    )

    dst_strides = torch.tensor(
        list(out.stride()),
        dtype=torch.int64,
        device=out.device,
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _unsqueeze_copy_kernel[grid](
        src,
        out,
        sizes,
        src_strides,
        dst_strides,
        n_elements,
        NDIM=src.dim(),
        INSERT_DIM=dim,
        BLOCK_SIZE=1024,
    )

    return out


def unsqueeze_copy(
    x: torch.Tensor,
    dim: int,
):
    """
    Insert a dimension of size 1 into the tensor layout and
    return a copied tensor.
    """

    logger.debug("GEMS UNSQUEEZE_COPY")

    if dim < 0:
        dim += x.dim() + 1

    if dim < 0 or dim > x.dim():
        raise IndexError(
            f"Dimension out of range "
            f"(expected in [{-x.dim() - 1}, {x.dim()}], got {dim})"
        )

    out_shape = list(x.shape)
    out_shape.insert(dim, 1)

    out = torch.empty(
        out_shape,
        dtype=x.dtype,
        device=x.device,
    )

    return _launch_unsqueeze_copy(
        x,
        dim,
        out,
    )


def unsqueeze_copy_out(
    x: torch.Tensor,
    dim: int,
    out: torch.Tensor,
):
    """
    out variant of unsqueeze_copy.
    """

    logger.debug("GEMS UNSQUEEZE_COPY_OUT")

    if dim < 0:
        dim += x.dim() + 1

    if dim < 0 or dim > x.dim():
        raise IndexError(
            f"Dimension out of range "
            f"(expected in [{-x.dim() - 1}, {x.dim()}], got {dim})"
        )

    expected_shape = list(x.shape)
    expected_shape.insert(dim, 1)

    if list(out.shape) != expected_shape:
        out.resize_(expected_shape)

    if out.dtype != x.dtype:
        raise RuntimeError("unsqueeze_copy_out: input and output dtype must match.")

    if out.device != x.device:
        raise RuntimeError(
            "unsqueeze_copy_out: input and output must be on the same device."
        )

    return _launch_unsqueeze_copy(
        x,
        dim,
        out,
    )
