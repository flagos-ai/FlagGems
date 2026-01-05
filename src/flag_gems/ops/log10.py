import logging
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "COMPLEX_TO_FLOAT")])
@triton.jit
def log10_func(x):
    inv_ln10 = 0.4342944819032518  # 1/ln(10)
    return tl.log(x.to(tl.float32)) * inv_ln10

def log10(A):
    logger.debug("GEMS LOG10")
    return log10_func(A)
