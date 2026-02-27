import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_tanh = tl_extra_shim.tanh
logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def mish_forward(x):
    x_fp32 = x.to(tl.float32)
    # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    softplus_x = tl.log(1.0 + tl.exp(x_fp32))
    y = x_fp32 * _tanh(softplus_x)
    return y


def mish(self):
    logger.debug("GEMS MISH")
    output = mish_forward(self)
    return output


def mish_(A):
    logger.debug("GEMS MISH_")
    out = mish_forward(A, out0=A)
    return out
