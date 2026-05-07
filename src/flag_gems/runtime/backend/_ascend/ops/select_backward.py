import logging

import torch

logger = logging.getLogger(__name__)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_ASCEND SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    out.zero_()
    out.select(dim, index).copy_(grad)
    return out
