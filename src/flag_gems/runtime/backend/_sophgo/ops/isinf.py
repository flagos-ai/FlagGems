import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isinf_func(x):
    # Simplified isinf implementation: direct comparison
    x_f32 = x.to(tl.float32)
    # Use constant comparison
    return (x_f32 == float("inf")) | (x_f32 == float("-inf"))


def isinf(input):
    logging.debug("GEMS ISINF SOPHGO")
    return isinf_func(input)
