import logging

import torch


logger = logging.getLogger(__name__)


def _normalize_dim(dim, ndim):
    if dim < 0:
        dim += ndim + 1

    if dim < 0 or dim > ndim:
        raise IndexError(
            f"Dimension out of range "
            f"(expected in [{-ndim-1}, {ndim}], got {dim})"
        )

    return dim


def unsqueeze_copy(
    x: torch.Tensor,
    dim: int,
):
    """
    Insert a size-1 dimension and return a copied tensor.
    """

    logger.debug("GEMS UNSQUEEZE_COPY")

    dim = _normalize_dim(dim, x.dim())

    view = x.unsqueeze(dim)

    out = torch.empty_like(view)

    out.copy_(view)

    return out


def unsqueeze_copy_out(
    x: torch.Tensor,
    dim: int,
    out: torch.Tensor,
):
    """
    out variant of unsqueeze_copy.
    """

    logger.debug("GEMS UNSQUEEZE_COPY_OUT")

    dim = _normalize_dim(dim, x.dim())

    view = x.unsqueeze(dim)

    if list(out.shape) != list(view.shape):
        out.resize_(tuple(view.shape))

    if out.dtype != x.dtype:
        raise RuntimeError(
            "unsqueeze_copy_out: input and output dtype must match."
        )

    if out.device != x.device:
        raise RuntimeError(
            "unsqueeze_copy_out: input and output must be on the same device."
        )

    out.copy_(view)

    return out
