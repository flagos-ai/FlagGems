import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[False, True], promotion_methods=[(0, 1, "BOOL_TO_LONG")])
@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent.to(tl.float32))


def pow_scalar(A, exponent):
    """
    Computes base^exponent where base is a scalar and exponent is a tensor.

    Uses FlagGems standard pointwise_dynamic for hardware compatibility.

    Args:
        A: Scalar base value
        exponent: Exponent tensor

    Returns:
        Output tensor with same shape as exponent
    """
    logger.debug("GEMS_ILUVATAR POW_SCALAR")
    return pow_func_scalar_tensor(A, exponent)


def pow_scalar_(A, exponent):
    """
    In-place version of pow_scalar.

    Args:
        A: Scalar base value
        exponent: Exponent tensor (modified in-place)

    Returns:
        The modified exponent tensor
    """
    logger.debug("GEMS_ILUVATAR POW_SCALAR_")
    return pow_func_scalar_tensor(A, exponent, out0=exponent)
