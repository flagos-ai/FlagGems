import logging

import torch
import triton

from ..utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def to_dtype_func(x):
    return x


def to_dtype(x, dtype, non_blocking=False, copy=False, memory_format=None):
    logger.debug("GEMS TO.DTYPE")
    if not copy and x.dtype == dtype:
        return x
    out = torch.empty_like(x, dtype=dtype, memory_format=memory_format)
    return to_dtype_func(x, out0=out)
