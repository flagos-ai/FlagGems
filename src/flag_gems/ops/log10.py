import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, COMPLEX_TO_FLOAT)])
@triton.jit
def log10_func(x):
    # log10(x) = ln(x) * log10(e)
    # Using high-precision constant: log10(e) = 1 / ln(10)
    LOG10_E = 0.43429448190325182765
    return tl.log(x.to(tl.float32)) * LOG10_E


def log10(A):
    """
    Computes the base-10 logarithm of the input tensor element-wise.
    
    Args:
        A: Input tensor
        
    Returns:
        Tensor with element-wise log10 values
    """
    logger.debug(GEMS LOG10)
    return log10_func(A)
