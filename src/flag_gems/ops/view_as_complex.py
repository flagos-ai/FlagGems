import logging

import torch

logger = logging.getLogger(__name__)


def view_as_complex(A: torch.Tensor):
    logger.debug("GEMS VIEW_AS_COMPLEX")
    if A.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "view_as_complex is only supported for float32 and float64 tensors"
        )
    if A.shape[-1] != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    return torch.complex(A[..., 0], A[..., 1])
