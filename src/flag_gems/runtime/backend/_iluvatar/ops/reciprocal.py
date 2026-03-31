import logging

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def reciprocal_kernel(x):
    return 1.0 / x


def reciprocal(input: torch.Tensor) -> torch.Tensor:
    """Compute reciprocal of input tensor: output = 1.0 / input"""
    logger.debug("GEMS RECIPROCAL - Iluvatar")
    # Empty tensor protection (Rule GR-030)
    if input.numel() == 0:
        return torch.empty_like(input)

    output = reciprocal_kernel(input)
    return output


def reciprocal_(input: torch.Tensor) -> torch.Tensor:
    """In-place reciprocal: input = 1.0 / input"""
    logger.debug("GEMS RECIPROCAL_ - Iluvatar in-place")
    # Empty tensor protection (Rule GR-030)
    if input.numel() == 0:
        return input

    # In-place variant
    out = reciprocal_kernel(input, out0=input)
    return out
