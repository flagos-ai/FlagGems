import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    logger.debug("GEMS DIAG_EMBED")

    rank = x.ndim + 1

    assert dim1 >= -rank and dim1 < rank, f"Invalid dim1: {dim1}"
    assert dim2 >= -rank and dim2 < rank, f"Invalid dim2: {dim2}"
    # convert from negative dims
    dim1 = dim1 % rank
    dim2 = dim2 % rank

    assert dim1 != dim2, "diagonal dimensions cannot be identical"

    # as per the docs, exchanging dims is equivalent to changing the sign of
    # offset
    if dim1 > dim2:
        offset = -offset
        dim1, dim2 = dim2, dim1

    # as per the docs, the size of last dim is placed at dim1 and dim2
    last_dim = x.size(-1) + abs(offset)

    y_shape = list(x.shape)
    y_shape.pop()
    y_shape.insert(dim1, last_dim)
    y_shape.insert(dim2, last_dim)

    y = torch.zeros(y_shape, dtype=x.dtype, device=x.device)
    y_diagonal_view = torch.diagonal(y, offset, dim1, dim2)
    copy_func.instantiate(x.ndim)(x, out0=y_diagonal_view)

    return y
