import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def arccosh_func(x):
    # arccosh(x) = log(x + sqrt(x - 1) * sqrt(x + 1))
    # Using float32 for intermediate computation to improve accuracy
    x32 = x.to(tl.float32)
    s1 = tl.sqrt(x32 - 1.0)
    s2 = tl.sqrt(x32 + 1.0)
    y32 = tl.log(x32 + s1 * s2)
    return y32.to(x.type.scalar)


def arccosh_(A):
    logger.debug("ILUVATAR GEMS ARCCOSH_")
    return arccosh_func(A, out0=A)
