import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def positive_kernel(x):
    return x


def positive(x):
    """
    Returns a direct reference to the input tensor (no-op / validation step).

    Args:
        x (torch.Tensor): Input tensor of any shape or dtype.

    Returns:
        torch.Tensor: The same tensor without modification.
    """
    logger.debug("GEMS POSITIVE FORWARD")
    return positive_kernel(x, out0=x)
