import logging

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def thnn_conv2d_impl(
    input, weight, kernel_size=0, bias=None, stride=1, padding=0, groups=1
):
    logger.debug("GEMS_SPACEMIT THNN_CONV2D")
    dilation = 1
    return conv2d(input, weight, bias, padding, stride, dilation, groups)


def thnn_conv2d(input, weight, kernel_size=0, bias=None, stride=1, padding=0, groups=1):
    return thnn_conv2d_impl(input, weight, kernel_size, bias, stride, padding, groups)
