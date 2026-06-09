import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS RSUB_TENSOR")
    return rsub_func(A, B, alpha)


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS RSUB_SCALAR")
    return rsub_func_tensor_scalar(A, B, alpha)
