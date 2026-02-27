import logging

from flag_gems.ops.conv1d import conv1d
from flag_gems.ops.conv2d import conv2d
from flag_gems.ops.conv3d import conv3d

logger = logging.getLogger(__name__)


def convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    """
    General convolution operation that dispatches to conv1d, conv2d, or conv3d
    based on input dimensions.

    Args:
        input: Input tensor of shape (N, C_in, *) where * is 1D, 2D, or 3D spatial dims
        weight: Weight tensor of shape (C_out, C_in/groups, *kernel_size)
        bias: Optional bias tensor of shape (C_out,)
        stride: Stride of the convolution
        padding: Padding added to input
        dilation: Spacing between kernel elements
        transposed: If True, performs transposed convolution (not supported)
        output_padding: Additional size for output shape (for transposed conv)
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor after convolution
    """
    logger.debug("GEMS CONVOLUTION")

    # Currently only support non-transposed convolution
    if transposed:
        raise NotImplementedError(
            "Transposed convolution is not supported in FlagGems. "
            "Please use the native PyTorch implementation."
        )

    # Determine convolution dimension based on input shape
    # input shape: (N, C, *spatial_dims)
    ndim = input.ndim
    spatial_ndim = ndim - 2  # Exclude batch and channel dimensions

    if spatial_ndim == 1:
        # 1D convolution
        return conv1d(
            input,
            weight,
            bias=bias,
            stride=stride[0] if isinstance(stride, (list, tuple)) else stride,
            padding=padding[0] if isinstance(padding, (list, tuple)) else padding,
            dilation=dilation[0] if isinstance(dilation, (list, tuple)) else dilation,
            groups=groups,
        )
    elif spatial_ndim == 2:
        # 2D convolution
        return conv2d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    elif spatial_ndim == 3:
        # 3D convolution
        return conv3d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    else:
        raise ValueError(
            f"Unsupported convolution dimension: {spatial_ndim}. "
            f"Only 1D, 2D, and 3D convolutions are supported. "
            f"Input shape: {input.shape}"
        )
