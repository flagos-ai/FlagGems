import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, 1, "COMPLEX_TO_FLOAT")])
@triton.jit
def float_power_func(x, exponent):
    return _pow(x.to(tl.float64), exponent.to(tl.float64))


def float_power(A, exponent):
    logger.debug("GEMS FLOAT_POWER")
    # float_power always promotes inputs to float64
    if isinstance(A, torch.Tensor):
        A = A.to(torch.float64)
    if isinstance(exponent, torch.Tensor):
        exponent = exponent.to(torch.float64)
    return float_power_func(A, exponent)
