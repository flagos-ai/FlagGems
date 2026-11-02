import logging
import math

import torch

logger = logging.getLogger(__name__)

_FAST_PATH_MAX_RANK = 64


def unsqueeze_(A: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place version of unsqueeze (zero-copy view operation).

    Mutates ``A`` itself: inserts a size-1 dimension into ``A``'s
    shape/strides in place, matching the semantics of ``torch.Tensor.unsqueeze_``.
    """
    logger.debug("GEMS_KUNLUNXIN UNSQUEEZE_")
    ndim = A.dim()
    d = dim if dim >= 0 else ndim + dim + 1
    if d < 0 or d > ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [0, {ndim}], "
            f"but got {d})"
        )
    if ndim > _FAST_PATH_MAX_RANK:
        # Generic path: identical to the baseline implementation
        # (``A.reshape`` + ``Tensor.set_``), kept for very high-rank inputs.
        new_shape = list(A.shape)
        new_shape.insert(d, 1)
        A.set_(A.reshape(new_shape))
        return A

    shape = list(A.shape)
    stride = list(A.stride())
    numel = A.numel()
    if numel == 0:
        # Zero-size tensor: match the generic implementation's product of the
        # non-degenerate following sizes (zero entries are skipped).
        insert_stride = 1
        for i in range(d, ndim):
            sz = A.shape[i]
            insert_stride *= sz if sz else 1
    elif d == 0:
        insert_stride = numel
    else:
        insert_stride = numel // math.prod(A.shape[:d])
    shape.insert(d, 1)
    stride.insert(d, insert_stride)
    A.as_strided_(shape, stride, A.storage_offset())
    return A
