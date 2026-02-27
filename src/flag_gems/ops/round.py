import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.triton_lang_helper import tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def round_func(x):
    # Use rint (round to nearest integer, ties to even) for banker's rounding
    # Convert to float32 for precision, then convert back
    return tl_extra_shim.rint(x.to(tl.float32)).to(x.dtype)


def round(A):
    logger.debug("GEMS ROUND")
    return round_func(A)


def round_(A):
    logger.debug("GEMS ROUND_")
    round_func(A, out0=A)
    return A
