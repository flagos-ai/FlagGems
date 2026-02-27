import logging

import torch

from flag_gems.ops.conv1d import conv1d
from flag_gems.ops.conv2d import conv2d
from flag_gems.ops.conv3d import conv3d

logger = logging.getLogger(__name__)


def _convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32,
):
    """
    General convolution function that dispatches to conv1d, conv2d, or conv3d
    based on input/weight dimensions.

    Args:
        input: Input tensor of shape (N, C_in, *spatial_dims)
        weight: Weight tensor of shape (C_out, C_in/groups, *kernel_dims)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride for each spatial dimension
        padding: Padding for each spatial dimension
        dilation: Dilation for each spatial dimension
        transposed: If True, perform transposed convolution
        output_padding: Additional padding for transposed convolution output
        groups: Number of groups for grouped convolution
        benchmark: cuDNN benchmark mode (ignored for Triton)
        deterministic: cuDNN deterministic mode (ignored for Triton)
        cudnn_enabled: Whether cuDNN is enabled (ignored for Triton)
        allow_tf32: Whether to allow TF32 (ignored for Triton)

    Returns:
        Output tensor
    """
    logger.debug("GEMS _CONVOLUTION")

    # Transposed convolution is not yet supported in FlagGems
    # Fall back to PyTorch for transposed convolution
    if transposed:
        logger.debug("GEMS _CONVOLUTION: Falling back to PyTorch for transposed conv")
        return torch._convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
            allow_tf32,
        )

    # Determine convolution dimension based on weight tensor
    # weight shape: (out_channels, in_channels/groups, *kernel_size)
    weight_ndim = weight.ndim
    spatial_dims = weight_ndim - 2  # Subtract batch and channel dimensions

    if spatial_dims == 1:
        # 1D convolution
        # stride, padding, dilation are lists of length 1
        s = stride[0] if isinstance(stride, (list, tuple)) else stride
        p = padding[0] if isinstance(padding, (list, tuple)) else padding
        d = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
        return conv1d(input, weight, bias, s, p, d, groups)

    elif spatial_dims == 2:
        # 2D convolution
        return conv2d(input, weight, bias, stride, padding, dilation, groups)

    elif spatial_dims == 3:
        # 3D convolution
        return conv3d(input, weight, bias, stride, padding, dilation, groups)

    else:
        raise ValueError(
            f"Unsupported convolution dimension: {spatial_dims}. "
            f"Only 1D, 2D, and 3D convolutions are supported."
        )
