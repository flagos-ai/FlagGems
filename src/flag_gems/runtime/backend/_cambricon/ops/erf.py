import logging

import triton
import triton.language as tl
from triton.language.extra.mlu.libdevice import fast_erf as _erf

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    output = _erf(x.to(tl.float32))
    return output


def erf(x):
    logger.debug("GEMS_CAMBRICON ERF")
    return erf_func(x)
