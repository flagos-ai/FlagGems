import logging

from flag_gems.ops.conv1d import conv1d
from flag_gems.ops.conv2d import conv2d
from flag_gems.ops.conv3d import conv3d

logger = logging.getLogger(__name__)


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """
    CUDNN convolution operation.

    This is a lower-level convolution operation that does not include bias.
    It supports 1D, 2D, and 3D convolutions based on the input dimensionality.

    Args:
        input: Input tensor of shape (N, C_in, *spatial_dims)
        weight: Weight tensor of shape (C_out, C_in/groups, *kernel_dims)
        padding: Padding for each spatial dimension
        stride: Stride for each spatial dimension
        dilation: Dilation for each spatial dimension
        groups: Number of groups for grouped convolution
        benchmark: cuDNN benchmark flag (ignored in Triton implementation)
        deterministic: cuDNN deterministic flag (ignored in Triton implementation)
        allow_tf32: Allow TF32 computation flag (ignored in Triton implementation)

    Returns:
        Output tensor after convolution
    """
    logger.debug("GEMS CUDNN_CONVOLUTION")

    ndim = input.ndim - 2

    # Extract values from lists if they are lists (cudnn_convolution receives lists)
    def extract_param(param, expected_len):
        if isinstance(param, (list, tuple)):
            if len(param) == expected_len:
                return param if expected_len > 1 else param[0]
            elif len(param) == 1:
                return param[0]
        return param

    if ndim == 1:
        # For 1D convolution, extract single values from lists
        stride_val = extract_param(stride, 1)
        padding_val = extract_param(padding, 1)
        dilation_val = extract_param(dilation, 1)
        return conv1d(
            input,
            weight,
            bias=None,
            stride=stride_val,
            padding=padding_val,
            dilation=dilation_val,
            groups=groups,
        )
    elif ndim == 2:
        return conv2d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    elif ndim == 3:
        return conv3d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        raise ValueError(
            f"cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )
