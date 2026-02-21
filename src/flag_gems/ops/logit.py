import logging
from typing import Optional

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def logit_kernel(x):
    xf = x.to(tl.float32)
    return tl.log(xf / (1.0 - xf)).to(x.dtype)


def logit(x, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT")
    if eps is not None:
        x = x.clamp(eps, 1.0 - eps)
    return logit_kernel(x)


def logit_(x, eps: Optional[float] = None):
    logger.debug("GEMS LOGIT_")
    if eps is not None:
        x.clamp_(eps, 1.0 - eps)
    return logit_kernel(x, out0=x)


def logit_out(x, eps: Optional[float] = None, *, out):
    logger.debug("GEMS LOGIT_OUT")
    if eps is not None:
        x = x.clamp(eps, 1.0 - eps)
    return logit_kernel(x, out0=out)
