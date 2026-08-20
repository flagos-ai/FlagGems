import logging

import torch

logger = logging.getLogger(__name__)

_composite_keyset = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_unsqueeze_view = torch.ops.aten.unsqueeze.default


def unsqueeze(A: torch.Tensor, dim: int) -> torch.Tensor:
    """Insert a dimension of size 1 at the specified position.

    This is a zero-copy view operation that adjusts shape and strides.
    No data is copied.

    Args:
        A: Input tensor
        dim: Index at which to insert the singleton dimension

    Returns:
        View of the input tensor with a dimension of size 1 inserted
    """
    logger.debug("GEMS UNSQUEEZE")

    ndim = A.dim()

    # Validate and normalize dim
    if dim < 0:
        dim = ndim + dim + 1

    if dim < 0 or dim > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [0, {ndim}], "
            f"but got {dim})"
        )

    new_shape = list(A.shape)
    new_shape.insert(dim, 1)

    # reshape delegates stride computation to PyTorch
    return A.reshape(new_shape)


def unsqueeze_(A: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place version of unsqueeze (zero-copy view operation).

    Mutates ``A`` itself: the new singleton dimension is inserted into
    ``A``'s shape/strides in place, so the change is visible through ``A``
    (and any alias of it), matching the semantics of
    ``torch.Tensor.unsqueeze_``.

    Builds a zero-copy singleton view through the C++ composite kernel and
    rebinds its metadata to the input tensor.
    """
    logger.debug("GEMS UNSQUEEZE_")
    view = _unsqueeze_view.redispatch(_composite_keyset, A, dim)
    return A.set_(view)
