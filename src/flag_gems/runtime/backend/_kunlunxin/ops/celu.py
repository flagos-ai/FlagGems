import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=True,  # TODO: Wait LLVM FIX
)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")])
@triton.jit
# celu another way: max(0, x) + alpha * (exp(min(0, x) / alpha) - 1), getting smaller instrs.
def celu_forward_kernel(x, alpha):
    inv_alpha = 1.0 / alpha

    pos_part = tl.maximum(0.0, x)

    neg_part_input = x - pos_part

    return pos_part + alpha * (tl.exp(neg_part_input * inv_alpha) - 1.0)


def celu(A, alpha=1.0):
    logger.debug("GEMS CELU")
    return celu_forward_kernel(A, alpha)


def celu_(A, alpha=1.0):
    logger.debug("GEMS CELU_")
    return celu_forward_kernel(A, alpha, out0=A)
