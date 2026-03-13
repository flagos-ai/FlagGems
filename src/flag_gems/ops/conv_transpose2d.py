import logging

import torch

logger = logging.getLogger(__name__)


def _pair(param, name):
    if isinstance(param, int):
        return (param, param)
    if isinstance(param, (list, tuple)) and len(param) == 2:
        return tuple(param)
    raise ValueError(f"Invalid {name}: {param}")


def conv_transpose2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS CONV_TRANSPOSE2D")

    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    output_padding = _pair(output_padding, "output_padding")
    dilation = _pair(dilation, "dilation")

    if groups <= 0:
        raise ValueError("groups must be a positive integer")

    if input.dim() == 3:
        input = input.unsqueeze(0)
        squeeze_output = True
    elif input.dim() == 4:
        squeeze_output = False
    else:
        raise ValueError("conv_transpose2d expects 3D or 4D input")

    output = torch.ops.aten.convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
    )
    return output.squeeze(0) if squeeze_output else output
