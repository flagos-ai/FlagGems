import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def celu_forward_kernel(x, alpha, inplace):
    return tl.where(
        x > 0,
        x,
        alpha * (tl.exp(x / alpha) - 1),
    )


def celu(A, alpha=1.0):
    logger.debug("GEMS_CAMBRICON CELU")
    return celu_forward_kernel(A, alpha, False)


def celu_(A, alpha=1.0):
    logger.debug("GEMS_CAMBRICON CELU_")
    return celu_forward_kernel(A, alpha, True, out0=A)
