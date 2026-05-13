import logging

import torch

logger = logging.getLogger(__name__)


def select_backward(grad, input_sizes, dim, index, out=None):
    logger.debug("GEMS_ASCEND SELECT_BACKWARD")
    dim = int(dim)
    index = int(index)
    sizes = list(input_sizes)
    ndim = len(sizes)

    assert dim >= -ndim and dim < ndim, "Invalid dim"
    dim %= ndim

    dim_size = sizes[dim]

    assert index >= -dim_size and index < dim_size, "Invalid index"
    index %= dim_size

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        assert tuple(out.shape) == tuple(sizes), "out shape mismatch"
        assert out.dtype == grad.dtype, "dtype mismatch"
        assert out.device == grad.device, "device mismatch"

    out.zero_()
    out.select(dim, index).copy_(grad)
    return out
