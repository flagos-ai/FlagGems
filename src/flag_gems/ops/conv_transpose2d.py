import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IntOrList = Union[int, List[int]]


def conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: _IntOrList = 1,
    padding: _IntOrList = 0,
    output_padding: _IntOrList = 0,
    groups: int = 1,
    dilation: _IntOrList = 1,
) -> torch.Tensor:
    """2-D transposed convolution (deconvolution).

    Args:
        input: 4-D input tensor (N, C_in, H, W).
        weight: Filter tensor (C_in, C_out/groups, kH, kW).
        bias: Optional bias of shape (C_out,).
        stride: Stride of the convolution.
        padding: Padding applied to input.
        output_padding: Additional size added to one side of the output.
        groups: Number of blocked connections.
        dilation: Spacing between kernel elements.

    Returns:
        Output tensor (N, C_out, H_out, W_out).
    """
    logger.debug("GEMS CONV_TRANSPOSE2D")
    return F.conv_transpose2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
