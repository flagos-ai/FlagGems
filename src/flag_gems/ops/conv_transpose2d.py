import logging

import torch

logger = logging.getLogger(__name__)


# conv_transpose2d: Transposed 2D convolution (also known as deconvolution).
# Applies a transposed convolution operator over an input image.
# Registered as FlagGems dispatch entry for framework integration.
# Delegates to PyTorch's native implementation which uses cuDNN on CUDA.
def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    logger.debug("GEMS CONV_TRANSPOSE2D")
    return torch.nn.functional.conv_transpose2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
