import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def celu_forward_kernel(x, alpha):
    return tl.where(
        x > 0,
        x,
        alpha * (tl.exp(x / alpha) - 1),
    )


def celu(A, alpha=1.0):
    logger.debug("GEMS CELU")
    return celu_forward_kernel(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS CELU_")
    return celu_forward_kernel(A, alpha, out0=A)
