import logging

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS_SPACEMIT DEPTHWISE")
    assert (
        input.ndim == 4
    ), f"Invalid input tensor: must be 4D, received shape {input.shape}"
    assert weight.shape[0] % input.shape[1] == 0, (
        f"Output channels must be a multiple of input channels, received "
        f"output channels {weight.shape[0]} and input channels {input.shape[1]}"
    )
    assert (
        weight.shape[1] == 1
    ), f"Input channels per group must be 1, received {weight.shape[1]}"
    groups = input.shape[1]
    return conv2d(input, weight, bias, stride, padding, dilation, groups)
