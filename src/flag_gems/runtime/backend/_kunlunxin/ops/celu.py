import logging

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseVectorization=True,  # TODO: Wait LLVM FIX
)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], config=config_
)
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
