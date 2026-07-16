import logging

import torch
import triton
import triton.language as tl


logger = logging.getLogger(__name__)


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    data = tl.load(
        src_ptr + offsets,
        mask=mask,
    )

    tl.store(
        dst_ptr + offsets,
        data,
        mask=mask,
    )


def _launch_copy(
    src: torch.Tensor,
    out: torch.Tensor,
):
    n_elements = src.numel()

    if n_elements == 0:
        return out

    grid = lambda meta: (
        triton.cdiv(
            n_elements,
            meta["BLOCK_SIZE"],
        ),
    )

    _copy_kernel[grid](
        src,
        out,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return out


def unsqueeze_copy(
    x: torch.Tensor,
    dim: int,
):
    """
    Insert a size-1 dimension and return a copied tensor.
    """

    logger.debug("GEMS UNSQUEEZE_COPY")

    if dim < 0:
        dim += x.dim() + 1

    if dim < 0 or dim > x.dim():
        raise IndexError(
            f"Dimension out of range "
            f"(expected in [{-x.dim()-1}, {x.dim()}], got {dim})"
        )

    out_shape = list(x.shape)
    out_shape.insert(dim, 1)

    out = torch.empty(
        out_shape,
        dtype=x.dtype,
        device=x.device,
    )

    return _launch_copy(
        x.contiguous(),
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
            f"(expected in [{-x.dim()-1}, {x.dim()}], got {dim})"
        )

    expected_shape = list(x.shape)
    expected_shape.insert(dim, 1)

    if list(out.shape) != expected_shape:
        out.resize_(expected_shape)

    if out.dtype != x.dtype:
        raise RuntimeError(
            "unsqueeze_copy_out: input and output dtype must match."
        )

    if out.device != x.device:
        raise RuntimeError(
            "unsqueeze_copy_out: input and output must be on the same device."
        )

    return _launch_copy(
        x.contiguous(),
        out,
    )
