import logging

from .conv2d import conv2d

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS DEPTHWISE")
    assert (
        input.ndim == 4
    ), "Invalid input tensor must be 4D, recevied shape {input.shape}"
    assert (
        weight.shape[0] % input.shape[1] == 0
    ), "Output channels must be multiple of input, recevied output {weught.shape[0], input {input.shape[0]}}"
    assert (
        weight.shape[1] == 1
    ), "input channels of per goups must be 1, recevied {weight.shape[1]}"
    groups = input.shape[1]
    return conv2d(input, weight, bias, stride, padding, dilation, groups)
