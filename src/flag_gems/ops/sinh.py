import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def sinh_kernel(x):
    """
    Compute sinh(x) for input x.

    The hyperbolic sine function: sinh(x) = (exp(x) - exp(-x)) / 2

    Args:
        x: Input tensor (will be converted to float32 for computation)

    Returns:
        Tensor with sinh(x) computed element-wise
    """
    x_fp32 = x.to(tl.float32)
    return (tl.exp(x_fp32) - tl.exp(-x_fp32)) / 2.0


def sinh(x):
    """
    Computes the hyperbolic sine (sinh) of each element in input.

    Args:
        x (Tensor): Input tensor with any real values

    Returns:
        Tensor: Output tensor with sinh values

    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> torch.sinh(x)
        tensor([0.0000, 1.1752, 3.6269])
    """
    logger.debug("GEMS SINH FORWARD")
    y = sinh_kernel(x)
    return y


def sinh_(x):
    """
    In-place version of sinh.

    Computes the hyperbolic sine of each element in input, modifying the tensor in-place.

    Args:
        x (Tensor): Input tensor (modified in-place)

    Returns:
        Tensor: The modified input tensor

    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> torch.sinh_(x)
        tensor([0.0000, 1.1752, 3.6269])
    """
    logger.debug("GEMS SINH_ INPLACE")
    sinh_kernel(x, out0=x)
    return x
