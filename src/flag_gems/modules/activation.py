import logging

import torch
import torch.nn as nn

import flag_gems

logger = logging.getLogger(__name__)

has_c_extension = False  # Disable C extension for now, as we have not implemented c++ wrapper for silu_and_mul yet.

__all__ = [
    "gems_silu_and_mul",
    "GemsSiluAndMul",
]


def gems_silu_and_mul(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS CUSTOM SILU_AND_MUL FORWARD")
    # TODO: Implement C++ wrapper for silu_and_mul
    return flag_gems.silu_and_mul(x, y)


class GemsSiluAndMul(nn.Module):
    """
    Fused Silu and Mul activation function.
    The function computes torch.mul(torch.nn.functional.silu(x), y) in a fused way.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return gems_silu_and_mul(x, y)
