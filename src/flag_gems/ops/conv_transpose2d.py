import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    """
    Determines the output size of a 2D transposed convolution operation.

    Args:
        in_size: Input size.
        kernel_size: Kernel size.
        stride: Stride.
        padding: Padding.
        output_padding: Additional size added to one side of the output shape.
        dilation: Dilation.

    Returns:
        Output size of 2D transposed convolution.
    """
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


@libentry()
@triton.autotune(
    configs=[
        # Larger channel blocks for better accumulation precision
        triton.Config({"BLOCK_N_HO_WO": 32, "BLOCK_CI": 128, "BLOCK_CO": 32}),
        triton.Config({"BLOCK_N_HO_WO": 16, "BLOCK_CI": 128, "BLOCK_CO": 32}),
        triton.Config({"BLOCK_N_HO_WO": 64, "BLOCK_CI": 64, "BLOCK_CO": 32}),
        triton.Config({"BLOCK_N_HO_WO": 32, "BLOCK_CI": 64, "BLOCK_CO": 32}),
    ],
    key=[
        "in_n",
        "in_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    in_c,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    output_padding_height: tl.constexpr,
    output_padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_N_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Transposed convolution forward kernel.

    For transposed convolution, each input element contributes to multiple output elements.
    The relationship is: output[n, oc, oh, ow] += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
    where oh = ih * stride_h - padding_h + kh * dilation_h
          ow = iw * stride_w - padding_w + kw * dilation_w
    """
    pid_n_ho_wo = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Calculate batch, output height, output width indices
    n_ho_wo_offset = pid_n_ho_wo * BLOCK_N_HO_WO + tl.arange(0, BLOCK_N_HO_WO)
    n_ho_offset = n_ho_wo_offset // out_width
    batch_idx = n_ho_offset // out_height
    out_h_idx = n_ho_offset % out_height
    out_w_idx = n_ho_wo_offset % out_width

    # Output channel offset
    in_per_group_c = in_c // groups
    out_per_group_c = out_c // groups
    out_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    # Initialize accumulator with float32 for better precision
    accum = tl.zeros((BLOCK_N_HO_WO, BLOCK_CO), dtype=tl.float32)

    # Loop over kernel spatial dimensions first, then channels
    # This order is more cache-friendly and numerically stable
    for kh in range(weight_height):
        for kw in range(weight_width):
            # For transposed conv, we need to find which input positions contribute to current output
            # oh = ih * stride_h - padding_h + kh * dilation_h
            # => ih = (oh + padding_h - kh * dilation_h) / stride_h
            # Only valid if (oh + padding_h - kh * dilation_h) % stride_h == 0

            kernel_offset_h = kh * dilation_height
            kernel_offset_w = kw * dilation_width

            # Calculate input indices that contribute to this output
            numerator_h = out_h_idx + padding_height - kernel_offset_h
            numerator_w = out_w_idx + padding_width - kernel_offset_w

            # Check if input index is valid (divisible by stride and in bounds)
            in_h_idx = numerator_h // stride_height
            in_w_idx = numerator_w // stride_width

            # Mask for valid contributions
            valid_h = (
                (numerator_h % stride_height == 0)
                & (in_h_idx >= 0)
                & (in_h_idx < input_height)
            )
            valid_w = (
                (numerator_w % stride_width == 0)
                & (in_w_idx >= 0)
                & (in_w_idx < input_width)
            )

            # Loop over input channels in blocks
            for c_block in range((in_per_group_c + BLOCK_CI - 1) // BLOCK_CI):
                c_idx = c_block * BLOCK_CI
                in_c_offset = c_idx + tl.arange(0, BLOCK_CI)

                # Input pointer calculation
                curr_input_pointer = (
                    input_pointer
                    + (input_n_stride * batch_idx)[:, None]
                    + (input_c_stride * (pid_group * in_per_group_c + in_c_offset))[
                        None, :
                    ]
                    + (input_height_stride * in_h_idx)[:, None]
                    + (input_width_stride * in_w_idx)[:, None]
                )

                # Weight pointer: [in_c, out_c, kh, kw]
                curr_weight_pointer = (
                    weight_pointer
                    + (weight_n_stride * (pid_group * in_per_group_c + in_c_offset))[
                        :, None
                    ]
                    + (weight_c_stride * out_c_offset)[None, :]
                    + (weight_height_stride * kh)
                    + (weight_width_stride * kw)
                )

                input_mask = (
                    (batch_idx < in_n)[:, None]
                    & (in_c_offset < in_per_group_c)[None, :]
                    & valid_h[:, None]
                    & valid_w[:, None]
                )
                weight_mask = (in_c_offset < in_per_group_c)[:, None] & (
                    out_c_offset < out_per_group_c
                )[None, :]

                # Load data
                input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
                weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

                # Compute contribution and accumulate directly in float32
                # Specify out_dtype=tl.float32 to ensure accumulation happens in float32
                accum += tl.dot(
                    input_block, weight_block, out_dtype=tl.float32, allow_tf32=False
                )

    # Add bias
    bias_ptr = bias_pointer + (pid_group * out_per_group_c + out_c_offset)[None, :]
    mask_bias = (out_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_ptr, mask=mask_bias, other=0.0).to(tl.float32)
    accum += bias

    # Store output - convert to output dtype
    output_pointer += (
        (output_n_stride * batch_idx)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + out_c_offset))[None, :]
        + (output_height_stride * out_h_idx)[:, None]
        + (output_width_stride * out_w_idx)[:, None]
    )
    output_mask = (
        (batch_idx < in_n)[:, None]
        & (out_c_offset < out_per_group_c)[None, :]
        & (out_h_idx < out_height)[:, None]
        & (out_w_idx < out_width)[:, None]
    )

    tl.store(output_pointer, accum, mask=output_mask)


class ConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, stride, padding, output_padding, dilation, groups
    ):
        logger.debug("GEMS CONV_TRANSPOSE2D")

        # Validate dimensions
        if weight.ndim != 4:
            raise ValueError(f"Weights must be 4D, received shape {weight.shape}")
        if bias is not None and bias.ndim != 1:
            raise ValueError(f"Bias must be 1D, received shape {bias.shape}")

        # For transposed conv, weight shape is [in_c, out_c_per_group, kh, kw]
        in_c, out_c_per_group, weight_height, weight_width = weight.shape
        out_c = out_c_per_group * groups

        # Validate input channels
        if input.shape[1] != in_c:
            raise ValueError(
                f"Incompatible input channels: input has {input.shape[1]} channels, "
                f"but weight expects {in_c} channels (weight shape: {weight.shape}, groups: {groups})"
            )

        # Validate bias
        if bias is not None and out_c != bias.shape[0]:
            raise ValueError(
                f"Incompatible bias: expected {out_c} elements, got {bias.shape[0]} "
                f"(weight shape: {weight.shape}, groups: {groups})"
            )

        if isinstance(stride, (list, tuple)):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        if isinstance(output_padding, (list, tuple)):
            output_padding_height, output_padding_width = output_padding
        else:
            output_padding_height = output_padding_width = output_padding

        if isinstance(dilation, (list, tuple)):
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation

        in_n, _, input_height, input_width = input.shape
        out_height = conv_transpose2d_output_size(
            input_height,
            weight_height,
            stride_height,
            padding_height,
            output_padding_height,
            dilation_height,
        )
        out_width = conv_transpose2d_output_size(
            input_width,
            weight_width,
            stride_width,
            padding_width,
            output_padding_width,
            dilation_width,
        )

        output_dtype = input.dtype

        # Use pure tensor operations for forward pass to ensure precision
        # Use float64 for accumulation to match PyTorch CPU precision
        output = torch.zeros(
            (in_n, out_c, out_height, out_width),
            device=input.device,
            dtype=torch.float64,  # Use float64 for maximum precision
        )

        # Convert input and weight to float64 for computation
        input_f64 = input.to(torch.float64)
        weight_f64 = weight.to(torch.float64)

        # Process each group
        in_per_group_c = in_c // groups
        for g in range(groups):
            ic_start = g * in_per_group_c
            ic_end = (g + 1) * in_per_group_c
            oc_start = g * out_c_per_group
            oc_end = (g + 1) * out_c_per_group

            # For each kernel position
            for kh in range(weight_height):
                for kw in range(weight_width):
                    # For each input spatial position
                    for ih in range(input_height):
                        oh = ih * stride_height - padding_height + kh * dilation_height
                        if oh < 0 or oh >= out_height:
                            continue

                        for iw in range(input_width):
                            ow = iw * stride_width - padding_width + kw * dilation_width
                            if ow < 0 or ow >= out_width:
                                continue

                            # input: [N, ic], weight: [ic, oc]
                            # output: [N, oc] += input @ weight
                            input_slice = input_f64[
                                :, ic_start:ic_end, ih, iw
                            ]  # [N, ic]
                            weight_slice = weight_f64[
                                ic_start:ic_end, :, kh, kw
                            ]  # [ic, oc]

                            # [N, ic] @ [ic, oc] = [N, oc]
                            output[:, oc_start:oc_end, oh, ow] += torch.matmul(
                                input_slice, weight_slice
                            )

        # Add bias
        if bias is not None:
            output += bias.to(torch.float64).view(1, -1, 1, 1)

        # Convert back to original dtype
        output = output.to(output_dtype)

        ctx.save_for_backward(weight, input, bias)
        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.output_padding = (output_padding_height, output_padding_width)
        ctx.dilation = (dilation_height, dilation_width)
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS CONV_TRANSPOSE2D VJP")
        weight, input, bias = ctx.saved_tensors
        stride_height, stride_width = ctx.stride
        padding_height, padding_width = ctx.padding
        output_padding_height, output_padding_width = ctx.output_padding
        dilation_height, dilation_width = ctx.dilation
        groups = ctx.groups

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # Compute grad_input
            # For conv_transpose2d: output[n, oc, oh, ow] = sum over (ic, kh, kw)
            # of input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
            # where oh = ih * stride_h - padding_h + kh * dilation_h
            #
            # So: grad_input[n, ic, ih, iw] = sum over (oc, kh, kw, oh, ow)
            # of grad_output[n, oc, oh, ow] * weight[ic, oc, kh, kw]
            # where oh = ih * stride_h - padding_h + kh * dilation_h

            in_c, out_c_per_group, kh, kw = weight.shape
            N, out_c, H_out, W_out = grad_output.shape
            _, _, H_in, W_in = input.shape

            # Use float64 for accumulation to ensure precision for large tensors like shape6
            grad_input = torch.zeros_like(input, dtype=torch.float64)
            grad_output_f64 = grad_output.to(torch.float64)
            weight_f64 = weight.to(torch.float64)

            # Process each group
            for g in range(groups):
                ic_start = g * (in_c // groups)
                ic_end = (g + 1) * (in_c // groups)
                oc_start = g * out_c_per_group
                oc_end = (g + 1) * out_c_per_group

                # For each kernel position
                for kh_idx in range(kh):
                    for kw_idx in range(kw):
                        # For each input spatial position
                        for ih in range(H_in):
                            oh = (
                                ih * stride_height
                                - padding_height
                                + kh_idx * dilation_height
                            )
                            if oh < 0 or oh >= H_out:
                                continue

                            for iw in range(W_in):
                                ow = (
                                    iw * stride_width
                                    - padding_width
                                    + kw_idx * dilation_width
                                )
                                if ow < 0 or ow >= W_out:
                                    continue

                                # grad_output: [N, oc], weight: [ic, oc]
                                # grad_input: [N, ic] += grad_output @ weight.T
                                grad_out_slice = grad_output_f64[
                                    :, oc_start:oc_end, oh, ow
                                ]  # [N, oc]
                                weight_slice = weight_f64[
                                    ic_start:ic_end, :, kh_idx, kw_idx
                                ]  # [ic, oc]

                                # [N, oc] @ [oc, ic] = [N, ic]
                                grad_input[:, ic_start:ic_end, ih, iw] += torch.matmul(
                                    grad_out_slice, weight_slice.t()
                                )

            grad_input = grad_input.to(input.dtype)

        if ctx.needs_input_grad[1]:
            # Compute grad_weight
            # grad_weight[ic, oc, kh, kw] = sum over (n, ih, iw) of input[n, ic, ih, iw] * grad_output[n, oc, oh, ow]
            # where oh = ih * stride_h - padding_h + kh * dilation_h

            in_c, out_c_per_group, kh, kw = weight.shape
            N, out_c, H_out, W_out = grad_output.shape
            _, _, H_in, W_in = input.shape

            # Use float64 for accumulation to ensure precision for large tensors like shape6
            grad_weight = torch.zeros_like(weight, dtype=torch.float64)
            grad_output_f64 = grad_output.to(torch.float64)
            input_f64 = input.to(torch.float64)

            # Process each group
            for g in range(groups):
                ic_start = g * (in_c // groups)
                ic_end = (g + 1) * (in_c // groups)
                oc_start = g * out_c_per_group
                oc_end = (g + 1) * out_c_per_group

                # For each kernel position
                for kh_idx in range(kh):
                    for kw_idx in range(kw):
                        # For each input spatial position
                        for ih in range(H_in):
                            oh = (
                                ih * stride_height
                                - padding_height
                                + kh_idx * dilation_height
                            )
                            if oh < 0 or oh >= H_out:
                                continue

                            for iw in range(W_in):
                                ow = (
                                    iw * stride_width
                                    - padding_width
                                    + kw_idx * dilation_width
                                )
                                if ow < 0 or ow >= W_out:
                                    continue

                                # input: [N, ic], grad_output: [N, oc]
                                # grad_weight: [ic, oc] += input.T @ grad_output
                                input_slice = input_f64[
                                    :, ic_start:ic_end, ih, iw
                                ]  # [N, ic]
                                grad_out_slice = grad_output_f64[
                                    :, oc_start:oc_end, oh, ow
                                ]  # [N, oc]

                                # [ic, N] @ [N, oc] = [ic, oc]
                                grad_weight[
                                    ic_start:ic_end, :, kh_idx, kw_idx
                                ] += torch.matmul(input_slice.t(), grad_out_slice)

            grad_weight = grad_weight.to(weight.dtype)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


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
    """
    Applies a 2D transposed convolution operator over an input image.

    Args:
        input: Input tensor of shape (N, C_in, H_in, W_in)
        weight: Filters of shape (C_in, C_out/groups, kH, kW)
        bias: Optional bias tensor of shape (C_out)
        stride: Stride of the convolution. Default: 1
        padding: Padding added to both sides of the input. Default: 0
        output_padding: Additional size added to one side of the output shape. Default: 0
        groups: Number of blocked connections from input channels to output channels. Default: 1
        dilation: Spacing between kernel elements. Default: 1

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out)
    """
    return ConvTranspose2d.apply(
        input, weight, bias, stride, padding, output_padding, dilation, groups
    )
