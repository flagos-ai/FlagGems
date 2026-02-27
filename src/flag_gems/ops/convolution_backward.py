import logging
from typing import List, Optional, Tuple

import torch
import triton

from flag_gems import runtime
from flag_gems.ops.conv2d import (
    conv2d_backward_kernel_weight,
    conv2d_forward_kernel,
    conv2d_output_size,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _conv2d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias_sizes: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    output_mask: List[bool],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute the backward pass for 2D convolution using Triton kernels.
    """
    device = input.device
    dtype = input.dtype

    stride_height, stride_width = stride[0], stride[1]
    padding_height, padding_width = padding[0], padding[1]
    dilation_height, dilation_width = dilation[0], dilation[1]

    in_n, in_c, input_height, input_width = input.shape
    out_c, weight_c, weight_height, weight_width = weight.shape

    out_height = grad_output.shape[2]
    out_width = grad_output.shape[3]

    grad_input = None
    grad_weight = None
    grad_bias = None

    # Compute grad_input if requested
    if output_mask[0]:
        # Compute gradient w.r.t. input using transposed convolution
        revert_padding_height = dilation_height * (weight_height - 1) - padding_height
        revert_padding_width = dilation_width * (weight_width - 1) - padding_width

        # Flip and transpose weight for backward pass
        revert_weight = weight.clone()
        revert_weight = torch.flip(revert_weight, dims=[2, 3]).contiguous()

        out_c_per_group = out_c // groups
        if groups != 1:
            revert_weight = revert_weight.reshape(
                groups, out_c_per_group, weight_c, weight_height, weight_width
            )
            revert_weight = revert_weight.transpose(1, 2)
            revert_weight = revert_weight.reshape(
                groups * weight_c, out_c_per_group, weight_height, weight_width
            ).contiguous()
        else:
            revert_weight = revert_weight.transpose(0, 1).contiguous()

        # Handle strided convolution by dilating grad_output
        new_out_height = out_height + (stride_height - 1) * (out_height - 1)
        new_out_width = out_width + (stride_width - 1) * (out_width - 1)

        if stride_height > 1 or stride_width > 1:
            new_out = torch.zeros(
                in_n,
                out_c,
                new_out_height,
                new_out_width,
                device=device,
                dtype=dtype,
            )
            # Copy grad_output to dilated positions
            for i in range(out_height):
                for j in range(out_width):
                    new_out[:, :, i * stride_height, j * stride_width] = grad_output[
                        :, :, i, j
                    ]
        else:
            new_out = grad_output

        # Allocate output for grad_input
        grad_input = torch.zeros(
            in_n,
            in_c,
            input_height,
            input_width,
            dtype=torch.float32,
            device=device,
        )

        # Use forward kernel with transposed weights to compute grad_input
        grid = lambda META: (
            triton.cdiv(
                in_n * input_height * input_width, META["BLOCK_NI_HO_WO"]
            ),
            triton.cdiv(weight_c, META["BLOCK_CO"]),
            groups,
        )

        bias_zero = torch.zeros(in_c, device=device, dtype=dtype)

        conv2d_forward_kernel[grid](
            new_out,
            revert_weight,
            grad_input,
            bias_zero,
            in_n,
            new_out_height,
            new_out_width,
            in_c,
            input_height,
            input_width,
            *new_out.stride(),
            *revert_weight.stride(),
            *grad_input.stride(),
            out_c_per_group,
            weight_height,
            weight_width,
            1,
            1,
            revert_padding_height,
            revert_padding_width,
            dilation_height,
            dilation_width,
            groups=groups,
        )

        grad_input = grad_input.to(dtype)

    # Compute grad_weight if requested
    if output_mask[1]:
        out_c_per_group = out_c // groups
        grad_weight = torch.zeros(
            out_c,
            weight_c,
            weight_height,
            weight_width,
            dtype=dtype,
            device=device,
        )

        grid_weight = lambda meta: (
            triton.cdiv(
                weight_c * weight_height * weight_width, meta["BLOCK_CI_HK_WK"]
            ),
            groups,
            triton.cdiv(out_c_per_group, meta["BLOCK_CO"]),
        )

        conv2d_backward_kernel_weight[grid_weight](
            input,
            grad_output,
            grad_weight,
            *input.stride(),
            *weight.stride(),
            *grad_output.stride(),
            input_height,
            input_width,
            weight_height,
            weight_width,
            weight_c,
            in_n,
            stride_height,
            stride_width,
            out_height,
            out_width,
            out_c_per_group,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
        )

    # Compute grad_bias if requested
    if output_mask[2] and bias_sizes is not None:
        grad_bias = grad_output.sum(dim=(0, 2, 3))

    return grad_input, grad_weight, grad_bias


def convolution_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias_sizes: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
    output_mask: List[bool],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute the backward pass for convolution operations.

    This function computes gradients with respect to input, weight, and bias
    for convolution operations based on the output_mask.

    Args:
        grad_output: Gradient of the output tensor
        input: Original input tensor
        weight: Convolution weight tensor
        bias_sizes: Size of bias (or None if no bias)
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        transposed: Whether this is a transposed convolution
        output_padding: Output padding for transposed convolution
        groups: Number of groups for grouped convolution
        output_mask: Boolean list [compute_grad_input, compute_grad_weight, compute_grad_bias]

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    logger.debug("GEMS CONVOLUTION_BACKWARD")

    # Determine convolution dimension from input shape
    # input shape: [N, C, ...spatial dims...]
    spatial_dims = input.ndim - 2

    # Currently only support 2D non-transposed convolution with Triton kernels
    # For other cases, fall back to native PyTorch implementation
    use_triton = (
        spatial_dims == 2
        and not transposed
        and all(op == 0 for op in output_padding)
        and input.is_cuda
        and grad_output.is_cuda
        and weight.is_cuda
    )

    if use_triton:
        return _conv2d_backward(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            groups,
            output_mask,
        )
    else:
        # Fall back to native PyTorch for unsupported cases
        # (1D, 3D, transposed convolution, etc.)
        return torch.ops.aten.convolution_backward.default(
            grad_output,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
