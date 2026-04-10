import logging

import torch

logger = logging.getLogger(__name__)


def _unpack_pair(value):
    """Extract a (height, width) pair from a scalar, list, or tuple."""
    if isinstance(value, (list, tuple)):
        return value[0], value[1]
    return value, value


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
    """Transposed 2D convolution (deconvolution).

    Implements conv_transpose2d by converting it to a regular convolution
    on a stride-inserted input with a flipped, channel-transposed weight.

    Args:
        input: (N, C_in, H_in, W_in) input tensor.
        weight: (C_in, C_out/groups, kH, kW) weight tensor.
        bias: Optional (C_out,) bias tensor.
        stride: Stride of the transposed convolution.
        padding: Padding of the transposed convolution.
        output_padding: Additional size added to one side of the output.
        groups: Number of groups for grouped convolution.
        dilation: Dilation factor.

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out).
    """
    logger.debug("GEMS CONV_TRANSPOSE2D")

    stride_h, stride_w = _unpack_pair(stride)
    padding_h, padding_w = _unpack_pair(padding)
    output_padding_h, output_padding_w = _unpack_pair(output_padding)
    dilation_h, dilation_w = _unpack_pair(dilation)

    in_n, c_in, h_in, w_in = input.shape
    weight_c_in, weight_c_out_per_group, kh, kw = weight.shape

    c_out = weight_c_out_per_group * groups

    # Step 1: Insert zeros (stride dilation) in the input
    needs_dilation = (
        stride_h > 1
        or stride_w > 1
        or output_padding_h > 0
        or output_padding_w > 0
    )

    if needs_dilation:
        dilated_h = h_in + (stride_h - 1) * (h_in - 1) + output_padding_h
        dilated_w = w_in + (stride_w - 1) * (w_in - 1) + output_padding_w
        dilated_input = torch.zeros(
            in_n, c_in, dilated_h, dilated_w,
            device=input.device,
            dtype=input.dtype,
        )
        dilated_input[:, :, ::stride_h, ::stride_w] = input
    else:
        dilated_input = input

    # Step 2: Flip kernel spatially and transpose channel dims
    flipped_weight = torch.flip(weight, dims=[2, 3])

    if groups != 1:
        c_in_per_group = c_in // groups
        reshaped = flipped_weight.reshape(
            groups, c_in_per_group, weight_c_out_per_group, kh, kw
        )
        transposed = reshaped.transpose(1, 2)
        conv_weight = transposed.reshape(
            c_out, c_in_per_group, kh, kw
        ).contiguous()
    else:
        conv_weight = flipped_weight.transpose(0, 1).contiguous()

    # Step 3: Compute effective padding for regular convolution
    eff_pad_h = dilation_h * (kh - 1) - padding_h
    eff_pad_w = dilation_w * (kw - 1) - padding_w

    # Step 4: Run regular convolution
    output = torch.conv2d(
        dilated_input,
        conv_weight,
        bias=bias,
        stride=(1, 1),
        padding=(eff_pad_h, eff_pad_w),
        dilation=(dilation_h, dilation_w),
        groups=groups,
    )

    return output
