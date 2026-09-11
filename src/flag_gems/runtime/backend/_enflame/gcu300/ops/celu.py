import logging

import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def celu_func(x, alpha):
    x_fp32 = x.to(tl.float32)
    alpha_fp32 = alpha.to(tl.float32)
    return tl.where(
        x_fp32 > 0, x_fp32, alpha_fp32 * (tl.exp(x_fp32 / alpha_fp32) - 1.0)
    ).to(x.dtype)


def celu(A, alpha=1.0):
    logger.debug("GEMS_ENFLAME CELU")
    return celu_func(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS_ENFLAME CELU_")
    celu_func(A, alpha, out0=A)
    return A
