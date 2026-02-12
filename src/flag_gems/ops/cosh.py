import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def cosh_kernel(x):
    """
    Compute cosh(x) for input x.

    The hyperbolic cosine function: cosh(x) = (exp(x) + exp(-x)) / 2

    Args:
        x: Input tensor (will be converted to float32 for computation)

    Returns:
        Tensor with cosh(x) computed element-wise
    """
    x_fp32 = x.to(tl.float32)
    return (tl.exp(x_fp32) + tl.exp(-x_fp32)) / 2.0


def cosh(x):
    """
    Computes the hyperbolic cosine (cosh) of each element in input.

    Args:
        x (Tensor): Input tensor with any real values

    Returns:
        Tensor: Output tensor with values >= 1

    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> torch.cosh(x)
        tensor([1.0000, 1.5431, 3.7622])
    """
    logger.debug("GEMS COSH FORWARD")
    y = cosh_kernel(x)
    return y


def cosh_(x):
    """
    In-place version of cosh.

    Computes the hyperbolic cosine of each element in input, modifying the tensor in-place.

    Args:
        x (Tensor): Input tensor (modified in-place)

    Returns:
        Tensor: The modified input tensor

    Example:
        >>> x = torch.tensor([0.0, 1.0, 2.0])
        >>> torch.cosh_(x)
        tensor([1.0000, 1.5431, 3.7622])
    """
    logger.debug("GEMS COSH_ INPLACE")
    cosh_kernel(x, out0=x)
    return x
