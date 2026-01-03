import logging
import triton
import triton.language as tl
import torch

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(
    promotion_method=[(0, "DEFAULT")],
)
@triton.jit
def acosh_forward_kernel(x)
    return tl.log(x + tl.sqrt(x * x - 1.0))

def acosh(input: torch.Tensor, *, out: torch.Tensor = None):
    """
    Returns a new tensor with the inverse hyperbolic cosine of the elements of input.

    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor

    Returns:
        Tensor: the output tensor with the inverse hyperbolic cosine values
    """
    result = acosh_forward_kernel(input)

    if out is not None:
        out.copy_(result)
        return out

    return output