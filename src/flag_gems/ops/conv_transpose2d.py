import logging

import torch
import triton

from flag_gems.ops.conv2d import conv2d_forward_kernel

logger = logging.getLogger(__name__)


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

    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride

    if isinstance(padding, (list, tuple)):
        padding_h, padding_w = padding
    else:
        padding_h = padding_w = padding

    if isinstance(output_padding, (list, tuple)):
        output_padding_h, output_padding_w = output_padding
    else:
        output_padding_h = output_padding_w = output_padding

    if isinstance(dilation, (list, tuple)):
        dilation_h, dilation_w = dilation
    else:
        dilation_h = dilation_w = dilation

    N, C_in, H_in, W_in = input.shape
    # weight layout: [C_in, C_out/groups, kH, kW]
    kH, kW = weight.shape[2], weight.shape[3]
    C_out_per_g = weight.shape[1]
    C_out = C_out_per_g * groups
    C_in_per_g = C_in // groups

    # Output spatial size
    H_out = (
        (H_in - 1) * stride_h
        - 2 * padding_h
        + dilation_h * (kH - 1)
        + output_padding_h
        + 1
    )
    W_out = (
        (W_in - 1) * stride_w
        - 2 * padding_w
        + dilation_w * (kW - 1)
        + output_padding_w
        + 1
    )

    # Revert weight: [C_in, C_out/g, kH, kW] → [C_out, C_in/g, kH, kW] (flipped)
    # This is the same transform used in Conv2d.backward for the input gradient
    if groups == 1:
        revert_weight = weight.flip([2, 3]).permute(1, 0, 2, 3).contiguous()
    else:
        revert_weight = weight.flip([2, 3])
        revert_weight = revert_weight.reshape(groups, C_in_per_g, C_out_per_g, kH, kW)
        revert_weight = revert_weight.transpose(1, 2)
        revert_weight = revert_weight.reshape(C_out, C_in_per_g, kH, kW).contiguous()

    # Dilate input: insert (stride-1) zeros between each input element
    # Using slice assignment (fast GPU scatter, avoids Python loops)
    if stride_h > 1 or stride_w > 1:
        H_dil = (H_in - 1) * stride_h + 1
        W_dil = (W_in - 1) * stride_w + 1
        x_dil = input.new_zeros(N, C_in, H_dil, W_dil)
        x_dil[:, :, ::stride_h, ::stride_w] = input
    else:
        x_dil = input
        H_dil = H_in
        W_dil = W_in

    # Effective padding for the equivalent conv2d on the dilated input
    revert_ph = dilation_h * (kH - 1) - padding_h
    revert_pw = dilation_w * (kW - 1) - padding_w

    output = torch.empty(N, C_out, H_out, W_out, device=input.device, dtype=input.dtype)

    if bias is None:
        bias_tensor = input.new_zeros(C_out)
    else:
        bias_tensor = bias

    # Call conv2d_forward_kernel: same kernel used by FlagGems conv2d forward pass.
    # Input  = dilated x_dil  [N, C_in, H_dil, W_dil]
    # Weight = revert_weight  [C_out, C_in/g, kH, kW]
    # Output = output         [N, C_out, H_out, W_out]
    # Stride = 1 (dilation already applied to input), padding = revert_ph/pw
    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_NI_HO_WO"]),
        triton.cdiv(C_out_per_g, META["BLOCK_CO"]),
        groups,
    )

    conv2d_forward_kernel[grid](
        x_dil,
        revert_weight,
        output,
        bias_tensor,
        N,
        H_dil,
        W_dil,
        C_out,
        H_out,
        W_out,
        *x_dil.stride(),
        *revert_weight.stride(),
        *output.stride(),
        C_in_per_g,  # weight_c: input channels per group
        kH,
        kW,
        1,  # stride_height = 1 (input already dilated)
        1,  # stride_width = 1
        revert_ph,
        revert_pw,
        dilation_h,
        dilation_w,
        groups=groups,
    )

    return output
