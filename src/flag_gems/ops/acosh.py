import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit()
def acosh_kernel(x):
    """
    Compute acosh(x) for input x.

    The inverse hyperbolic cosine function returns values >= 0 for input values >= 1.
    For x < 1, the result is NaN.

    Args:
        x: Input tensor (will be converted to float32 for computation)

    Returns:
        Tensor with acosh(x) computed element-wise
    """
    return tl.log(x.to(tl.float32) + tl.sqrt(x.to(tl.float32) * x.to(tl.float32) - 1.0))


def acosh(x):
    """
    Computes the inverse hyperbolic cosine (acosh) of each element in input.

    Args:
        x (Tensor): Input tensor with values >= 1

    Returns:
        Tensor: Output tensor with values >= 0

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> torch.acosh(x)
        tensor([0.0000, 1.3170, 1.7627])
    """
    logger.debug("GEMS ACOSH FORWARD")
    y = acosh_kernel(x)
    return y


def acosh_(x):
    """
    In-place version of acosh.

    Computes the inverse hyperbolic cosine of each element in input, modifying the tensor in-place.

    Args:
        x (Tensor): Input tensor with values >= 1 (modified in-place)

    Returns:
        Tensor: The modified input tensor

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> torch.acosh_(x)
        tensor([0.0000, 1.3170, 1.7627])
    """
    logger.debug("GEMS ACOSH_ INPLACE")
    acosh_kernel(x, out0=x)
    return x
