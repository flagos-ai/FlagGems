import logging

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def conv1d(input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV1D")

    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (padding_width, 0),
        (stride_width, 1),
        dilation,
        groups,
    ).squeeze(-1)
