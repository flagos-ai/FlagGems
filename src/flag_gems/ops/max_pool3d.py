"""
MaxPool3d operator implementation for FlagGems.

This module provides the 3D max pooling operation using Triton kernels.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


def max_pool3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool = False,
) -> int:
    """
    Calculate output size for one spatial dimension.

    Args:
        in_size: Input size for the dimension
        kernel_size: Kernel size for the dimension
        stride: Stride for the dimension
        padding: Padding for the dimension
        dilation: Dilation for the dimension
        ceil_mode: Whether to use ceil instead of floor

    Returns:
        Output size for the dimension
    """
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    numerator = in_size + 2 * padding - effective_kernel_size
    if ceil_mode:
        output_size = (numerator + stride - 1) // stride + 1
        # PyTorch-compatible adjustment for ceil_mode
        if (output_size - 1) * stride >= in_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    return max(0, output_size)


def _parse_pool_params_3d(kernel_size, stride, padding, dilation):
    """
    Parse and validate pooling parameters for 3D pooling.

    Args:
        kernel_size: Size of the pooling window (int or tuple of 3 ints)
        stride: Stride of the pooling (int or tuple of 3 ints, or None)
        padding: Padding added to all sides (int or tuple of 3 ints)
        dilation: Spacing between kernel elements (int or tuple of 3 ints)

    Returns:
        Tuple of 12 values: (kernel_d, kernel_h, kernel_w,
                            stride_d, stride_h, stride_w,
                            padding_d, padding_h, padding_w,
                            dilation_d, dilation_h, dilation_w)

    Raises:
        ValueError: If parameters are invalid
    """

    def _parse_param(param, name, default=None):
        """Helper to parse a single parameter."""
        if param is None:
            return default
        # Handle empty list/tuple (PyTorch may pass [] for unspecified params)
        if isinstance(param, (list, tuple)) and len(param) == 0:
            return default
        if isinstance(param, int):
            return param, param, param
        if isinstance(param, (list, tuple)) and len(param) == 3:
            return param
        raise ValueError(
            f"Invalid {name}: {param}. Expected int or tuple/list of 3 ints."
        )

    # Parse kernel_size
    kernel = _parse_param(kernel_size, "kernel_size")
    if kernel is None:
        raise ValueError("kernel_size must be specified")
    kernel_d, kernel_h, kernel_w = kernel

    # Parse stride (defaults to kernel_size if None)
    stride_default = (kernel_d, kernel_h, kernel_w)
    stride = _parse_param(stride, "stride", default=stride_default)
    stride_d, stride_h, stride_w = stride

    # Parse padding
    padding = _parse_param(padding, "padding", default=(0, 0, 0))
    padding_d, padding_h, padding_w = padding

    # Parse dilation
    dilation = _parse_param(dilation, "dilation", default=(1, 1, 1))
    dilation_d, dilation_h, dilation_w = dilation

    # Validate parameters
    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, but got ({stride_d}, {stride_h}, {stride_w})"
        )

    if padding_d < 0 or padding_h < 0 or padding_w < 0:
        raise ValueError(
            f"padding must be non-negative, but got ({padding_d}, {padding_h}, {padding_w})"
        )

    if dilation_d <= 0 or dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, but got ({dilation_d}, {dilation_h}, {dilation_w})"
        )

    return (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    )


def _prepare_5d_input(t):
    """
    Prepare input tensor to be 5D.

    Args:
        t: Input tensor (4D or 5D)

    Returns:
        Tuple of (tensor_5d, was_squeezed) where:
            - tensor_5d: 5D tensor (N, C, D, H, W)
            - was_squeezed: True if input was 4D and batch dimension was added

    Raises:
        ValueError: If input is not 4D or 5D
    """
    if t.dim() == 5:
        return t, False
    if t.dim() == 4:
        return t.unsqueeze(0), True  # add N=1
    raise ValueError(
        "input for max_pool3d must be 4D (C,D,H,W) or 5D (N,C,D,H,W), "
        f"but got {t.dim()}D tensor"
    )


def max_pool3d_im2col(
    input, kernel_size, stride, padding, dilation, return_indices=False
):
    """
    3D max pooling using fused Triton kernel (unfold + max in one pass).

    This method is faster for large kernels as it avoids intermediate storage.

    Args:
        input: Input tensor (N, C, D, H, W)
        kernel_size: Tuple of (kD, kH, kW)
        stride: Tuple of (sD, sH, sW)
        padding: Tuple of (pD, pH, pW)
        dilation: Tuple of (dD, dH, dW)
        return_indices: Whether to return indices

    Returns:
        output: Pooled tensor
        indices: (optional) Indices of max values
    """
    N, C, in_d, in_h, in_w = input.shape
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    # Calculate output sizes
    out_d = (in_d + 2 * pD - (kD - 1) * dD - 1) // sD + 1
    out_h = (in_h + 2 * pH - (kH - 1) * dH - 1) // sH + 1
    out_w = (in_w + 2 * pW - (kW - 1) * dW - 1) // sW + 1

    # Grid: each program handles one output position
    num_outputs = N * C * out_d * out_h * out_w
    grid = (num_outputs,)

    if not return_indices:
        # Use fused kernel for better performance (unfold + max in one pass)
        output = torch.empty(
            (N, C, out_d, out_h, out_w), dtype=input.dtype, device=input.device
        )

        max_pool3d_fused_kernel[grid](
            input,
            output,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            input.stride(4),
            N,
            C,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kD,
            kH,
            kW,
            sD,
            sH,
            sW,
            pD,
            pH,
            pW,
            dD,
            dH,
            dW,
        )
        return output
    else:
        # Use fused kernel with indices tracking
        output = torch.empty(
            (N, C, out_d, out_h, out_w), dtype=input.dtype, device=input.device
        )
        indices = torch.empty(
            (N, C, out_d, out_h, out_w), dtype=torch.int64, device=input.device
        )

        max_pool3d_fused_kernel_with_indices[grid](
            input,
            output,
            indices,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            input.stride(4),
            N,
            C,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kD,
            kH,
            kW,
            sD,
            sH,
            sW,
            pD,
            pH,
            pW,
            dD,
            dH,
            dW,
        )
        return output, indices


# Fused kernel: unfold + max reduction in one pass
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_fused_kernel"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kD",
        "kH",
        "kW",
    ],
)
@triton.jit
def max_pool3d_fused_kernel(
    input_ptr,
    output_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Tensor dimensions
    N: tl.constexpr,
    C: tl.constexpr,
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    # Pooling parameters
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    sD: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pD: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dD: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
):
    """
    Fused 3D max pooling kernel - performs unfold and max reduction in one pass.

    Each program instance handles one output position, directly reading from the
    kernel window in input and computing the max value. This avoids storing
    intermediate patches like PyTorch's unfold would require.
    """
    # Get the flat output position for this thread
    pid = tl.program_id(0)
    total_out = out_d * out_h * out_w

    # Decompose into NC and spatial position
    nc = pid // total_out
    spatial = pid % total_out

    # Decompose spatial position into 3D
    out_d_idx = spatial // (out_h * out_w)
    out_h_idx = (spatial // out_w) % out_h
    out_w_idx = spatial % out_w

    # Calculate input starting positions for this output
    d_start = out_d_idx * sD - pD
    h_start = out_h_idx * sH - pH
    w_start = out_w_idx * sW - pW

    # Initialize accumulator
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val = min_val

    # Unfold + Max: iterate through kernel window and find max
    for kd in tl.static_range(0, kD):
        for kh in tl.static_range(0, kH):
            for kw in tl.static_range(0, kW):
                d_in = d_start + kd * dD
                h_in = h_start + kh * dH
                w_in = w_start + kw * dW

                # Check bounds
                in_bounds = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                if in_bounds:
                    # Calculate input offset considering strides
                    input_offset = (
                        nc * in_d * in_h * in_w
                        + d_in * in_stride_d
                        + h_in * in_stride_h
                        + w_in * in_stride_w
                    )

                    # Load from input (this is the "unfold" step)
                    val = tl.load(input_ptr + input_offset)

                    # Update max (this is the "reduction" step)
                    max_val = tl.maximum(max_val, val)

    # Calculate flat output offset
    out_offset = pid

    # Store output
    tl.store(output_ptr + out_offset, max_val)


# Fused kernel: unfold + max reduction + argmax in one pass (for return_indices=True)
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_fused_kernel_with_indices"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kD",
        "kH",
        "kW",
    ],
)
@triton.jit
def max_pool3d_fused_kernel_with_indices(
    input_ptr,
    output_ptr,
    indices_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Tensor dimensions
    N: tl.constexpr,
    C: tl.constexpr,
    in_d: tl.constexpr,
    in_h: tl.constexpr,
    in_w: tl.constexpr,
    out_d: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    # Pooling parameters
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    sD: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pD: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dD: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
):
    """
    Fused 3D max pooling kernel with indices tracking.

    Each program instance handles one output position, directly reading from the
    kernel window in input, computing both the max value and its flat index.
    """
    # Get the flat output position for this thread
    pid = tl.program_id(0)
    total_out = out_d * out_h * out_w

    # Decompose into NC and spatial position
    nc = pid // total_out
    spatial = pid % total_out

    # Decompose nc into batch and channel
    n_idx = nc // C
    c_idx = nc % C

    # Decompose spatial position into 3D
    out_d_idx = spatial // (out_h * out_w)
    out_h_idx = (spatial // out_w) % out_h
    out_w_idx = spatial % out_w

    # Calculate input starting positions for this output
    d_start = out_d_idx * sD - pD
    h_start = out_h_idx * sH - pH
    w_start = out_w_idx * sW - pW

    # Initialize accumulators
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val = min_val
    max_idx = -1

    # Base pointer for input (skip to correct batch and channel)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Unfold + Max + Argmax: iterate through kernel window
    for kd in tl.static_range(0, kD):
        for kh in tl.static_range(0, kH):
            for kw in tl.static_range(0, kW):
                d_in = d_start + kd * dD
                h_in = h_start + kh * dH
                w_in = w_start + kw * dW

                # Check bounds
                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                # Calculate input offset
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )

                # Load from input with masking
                val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )

                # Compute flat index (PyTorch format: flat index in spatial dimensions)
                flat_idx = d_in * in_h * in_w + h_in * in_w + w_in

                # Update max and argmax
                is_new_max = val > max_val
                max_val = tl.where(is_new_max, val, max_val)
                max_idx = tl.where(is_new_max & in_mask, flat_idx, max_idx)

    # Calculate flat output offset
    out_offset = pid

    # Store output and indices
    tl.store(output_ptr + out_offset, max_val)
    tl.store(indices_ptr + out_offset, max_idx)


# Optimized kernel for large kernels (no indices tracking)
# Uses output-centric approach like the standard kernel but optimized for register pressure
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_forward_kernel_large_no_indices"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
        "padding_d",
        "padding_h",
        "padding_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel_large_no_indices(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    3D max pooling forward kernel optimized for large kernels (no indices).

    Uses output-centric approach with small blocks to reduce register pressure.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Decompose program_id into spatial block indices
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    dhw_block_idx = pid_dhw
    w_block_idx = dhw_block_idx % num_w_blocks
    h_block_idx = (dhw_block_idx // num_w_blocks) % num_h_blocks
    d_block_idx = dhw_block_idx // (num_w_blocks * num_h_blocks)

    # Decompose nc into batch and channel
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Compute output offsets for this block
    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Reshape to 3D for broadcasting
    d_out_offsets_3d = d_out_offsets[:, None, None]
    h_out_offsets_3d = h_out_offsets[None, :, None]
    w_out_offsets_3d = w_out_offsets[None, None, :]

    # Initialize accumulator
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    # Base pointer for input (skip to correct batch and channel)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Iterate over kernel
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                # Compute corresponding input positions
                d_in = d_out_offsets_3d * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets_3d * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets_3d * stride_w - padding_w + kw * dilation_w

                # Mask for valid input positions
                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                # Load input values
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )

                # Update max values only (no indices tracking)
                max_val_acc = tl.maximum(max_val_acc, current_val)

    # Store output
    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w

    output_block_ptr = (
        out_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )

    out_mask = (
        (d_out_offsets_3d < out_d)
        & (h_out_offsets_3d < out_h)
        & (w_out_offsets_3d < out_w)
    )

    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_forward_kernel_large"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
        "padding_d",
        "padding_h",
        "padding_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel_large(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    3D max pooling forward kernel optimized for large kernels.

    Uses smaller block sizes to reduce register pressure when kernel_size is large.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Decompose program_id into spatial block indices
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    dhw_block_idx = pid_dhw
    w_block_idx = dhw_block_idx % num_w_blocks
    h_block_idx = (dhw_block_idx // num_w_blocks) % num_h_blocks
    d_block_idx = dhw_block_idx // (num_w_blocks * num_h_blocks)

    # Decompose nc into batch and channel
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Compute output offsets for this block
    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Reshape to 3D for broadcasting
    d_out_offsets_3d = d_out_offsets[:, None, None]
    h_out_offsets_3d = h_out_offsets[None, :, None]
    w_out_offsets_3d = w_out_offsets[None, None, :]

    # Initialize accumulators
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    # Base pointer for input (skip to correct batch and channel)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Iterate over kernel
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                # Compute corresponding input positions
                d_in = d_out_offsets_3d * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets_3d * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets_3d * stride_w - padding_w + kw * dilation_w

                # Mask for valid input positions
                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                # Load input values
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )

                # Compute flat index for this position
                current_idx = d_in * in_h * in_w + h_in * in_w + w_in

                # Update max values and indices
                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    # Store output
    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_d * out_h * out_w

    output_block_ptr = (
        out_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )
    indices_block_ptr = (
        indices_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )

    out_mask = (
        (d_out_offsets_3d < out_d)
        & (h_out_offsets_3d < out_h)
        & (w_out_offsets_3d < out_w)
    )

    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_forward_kernel"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
        "padding_d",
        "padding_h",
        "padding_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    3D max pooling forward kernel.

    Each program instance processes one channel of one batch element,
    handling a tile of output positions of size BLOCK_D x BLOCK_H x BLOCK_W.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Decompose program_id into spatial block indices
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    dhw_block_idx = pid_dhw
    w_block_idx = dhw_block_idx % num_w_blocks
    h_block_idx = (dhw_block_idx // num_w_blocks) % num_h_blocks
    d_block_idx = dhw_block_idx // (num_w_blocks * num_h_blocks)

    # Decompose nc into batch and channel
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Compute output offsets for this block
    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Reshape to 3D for broadcasting
    d_out_offsets_3d = d_out_offsets[:, None, None]
    h_out_offsets_3d = h_out_offsets[None, :, None]
    w_out_offsets_3d = w_out_offsets[None, None, :]

    # Initialize accumulators
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), min_val, dtype=dtype)
    max_idx_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), -1, dtype=tl.int64)

    # Base pointer for input (skip to correct batch and channel)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Iterate over kernel
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                # Compute corresponding input positions
                d_in = d_out_offsets_3d * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets_3d * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets_3d * stride_w - padding_w + kw * dilation_w

                # Mask for valid input positions
                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                # Load input values
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )

                # Compute flat index for this position
                current_idx = d_in * in_h * in_w + h_in * in_w + w_in

                # Update max values and indices
                is_new_max = current_val > max_val_acc
                max_val_acc = tl.where(is_new_max, current_val, max_val_acc)
                max_idx_acc = tl.where(is_new_max & in_mask, current_idx, max_idx_acc)

    # Store output - reuse offsets calculated above (no redundant recalculation)
    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w
    indices_base_ptr = indices_ptr + pid_nc * out_d * out_h * out_w

    # Reuse d_out_offsets_3d, h_out_offsets_3d, w_out_offsets_3d from lines 256-258
    output_block_ptr = (
        out_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )
    indices_block_ptr = (
        indices_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )

    out_mask = (
        (d_out_offsets_3d < out_d)
        & (h_out_offsets_3d < out_h)
        & (w_out_offsets_3d < out_w)
    )

    tl.store(output_block_ptr, max_val_acc, mask=out_mask)
    tl.store(indices_block_ptr, max_idx_acc, mask=out_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max_pool3d_forward_kernel_no_indices"),
    key=[
        "out_d",
        "out_h",
        "out_w",
        "kernel_d",
        "kernel_h",
        "kernel_w",
        "stride_d",
        "stride_h",
        "stride_w",
        "padding_d",
        "padding_h",
        "padding_w",
    ],
)
@triton.jit
def max_pool3d_forward_kernel_no_indices(
    input_ptr,
    output_ptr,
    # Input tensor strides
    in_stride_n,
    in_stride_c,
    in_stride_d,
    in_stride_h,
    in_stride_w,
    # Input/Output shapes
    in_c,
    in_d,
    in_h,
    in_w,
    out_d,
    out_h,
    out_w,
    # Pooling parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_d: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_d: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    # Meta-parameters for tiling
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    3D max pooling forward kernel (optimized without indices tracking).

    This kernel is faster and uses less memory when return_indices=False.
    """
    pid_nc = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Decompose program_id into spatial block indices
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    dhw_block_idx = pid_dhw
    w_block_idx = dhw_block_idx % num_w_blocks
    h_block_idx = (dhw_block_idx // num_w_blocks) % num_h_blocks
    d_block_idx = dhw_block_idx // (num_w_blocks * num_h_blocks)

    # Decompose nc into batch and channel
    n_idx = pid_nc // in_c
    c_idx = pid_nc % in_c

    # Compute output offsets for this block (calculated once)
    d_out_offsets = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    # Reshape to 3D for broadcasting
    d_out_offsets_3d = d_out_offsets[:, None, None]
    h_out_offsets_3d = h_out_offsets[None, :, None]
    w_out_offsets_3d = w_out_offsets[None, None, :]

    # Initialize accumulator
    dtype = input_ptr.type.element_ty
    min_val = get_dtype_min(dtype)
    max_val_acc = tl.full((BLOCK_D, BLOCK_H, BLOCK_W), min_val, dtype=dtype)

    # Base pointer for input (skip to correct batch and channel)
    input_base_ptr = input_ptr + n_idx * in_stride_n + c_idx * in_stride_c

    # Iterate over kernel
    for kd in tl.static_range(0, kernel_d):
        for kh in tl.static_range(0, kernel_h):
            for kw in tl.static_range(0, kernel_w):
                # Compute corresponding input positions
                d_in = d_out_offsets_3d * stride_d - padding_d + kd * dilation_d
                h_in = h_out_offsets_3d * stride_h - padding_h + kh * dilation_h
                w_in = w_out_offsets_3d * stride_w - padding_w + kw * dilation_w

                # Mask for valid input positions
                in_mask = (
                    (d_in >= 0)
                    & (d_in < in_d)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                # Load input values
                input_offset = (
                    d_in * in_stride_d + h_in * in_stride_h + w_in * in_stride_w
                )
                current_val = tl.load(
                    input_base_ptr + input_offset, mask=in_mask, other=min_val
                )

                # Update max values only (no indices tracking)
                max_val_acc = tl.maximum(max_val_acc, current_val)

    # Store output - reuse offsets calculated above (no redundant recalculation)
    out_base_ptr = output_ptr + pid_nc * out_d * out_h * out_w

    output_block_ptr = (
        out_base_ptr
        + d_out_offsets_3d * out_h * out_w
        + h_out_offsets_3d * out_w
        + w_out_offsets_3d
    )

    out_mask = (
        (d_out_offsets_3d < out_d)
        & (h_out_offsets_3d < out_h)
        & (w_out_offsets_3d < out_w)
    )

    tl.store(output_block_ptr, max_val_acc, mask=out_mask)


def max_pool3d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    3D max pooling operation.

    Args:
        input: Input tensor of shape (N, C, D, H, W) or (C, D, H, W)
        kernel_size: Size of the pooling window. Can be a single number
                    or a tuple (kD, kH, kW)
        stride: Stride of the pooling. Can be a single number or a tuple
                (sD, sH, sW). Default: kernel_size
        padding: Padding added to all sides. Can be a single number or a
                 tuple (padD, padH, padW). Default: 0
        dilation: Spacing between kernel elements. Can be a single number
                  or a tuple (dD, dH, dW). Default: 1
        ceil_mode: If True, will use ceil instead of floor to compute the
                   output shape. Default: False
        return_indices: If True, will return the argmax along with the max
                       values. Useful for max_unpool3d later. Default: False

    Returns:
        If return_indices is False: output tensor of shape (N, C, D_out, H_out, W_out)
        If return_indices is True: tuple of (output, indices) tensors

    Note:
        This implementation uses two specialized kernels for optimization:
        - With indices tracking: when return_indices=True
        - Without indices tracking: when return_indices=False (faster, less memory)
    """
    logger.debug("GEMS MAX_POOL3D FORWARD")

    # Prepare input (handle 4D/5D)
    input_5d, was_squeezed = _prepare_5d_input(input)
    input_5d = input_5d.contiguous()

    # Parse parameters
    params = _parse_pool_params_3d(kernel_size, stride, padding, dilation)
    (
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
    ) = params

    # Get input shape
    in_n, in_c, in_d, in_h, in_w = input_5d.shape

    # Calculate output sizes
    out_d = max_pool3d_output_size(
        in_d, kernel_d, stride_d, padding_d, dilation_d, ceil_mode
    )
    out_h = max_pool3d_output_size(
        in_h, kernel_h, stride_h, padding_h, dilation_h, ceil_mode
    )
    out_w = max_pool3d_output_size(
        in_w, kernel_w, stride_w, padding_w, dilation_w, ceil_mode
    )

    # Check if we should use fused approach for large kernels
    # Fused kernel (one output per program) is better for large kernels due to:
    # - Better memory coalescing
    # - Lower register pressure
    # - Simpler control flow
    # Only use for truly large kernels (7x7x7=343 or larger)
    # Standard tiled kernel is more efficient for smaller kernels
    kernel_product = kernel_d * kernel_h * kernel_w
    use_fused_kernel = kernel_product >= 343  # Only for kernels >= 7x7x7

    # Create output tensor
    output = torch.empty(
        (in_n, in_c, out_d, out_h, out_w),
        device=input_5d.device,
        dtype=input_5d.dtype,
    )

    # Create indices tensor ONLY if needed
    if return_indices:
        indices = torch.empty(
            (in_n, in_c, out_d, out_h, out_w),
            device=input_5d.device,
            dtype=torch.int64,
        )

    # Handle empty tensor case
    if output.numel() == 0:
        if return_indices:
            if was_squeezed:
                return output.squeeze(0), indices.squeeze(0)
            return output, indices
        else:
            if was_squeezed:
                return output.squeeze(0)
            return output

    # Use fused kernel approach for large kernels
    # Each program handles one output position - better memory coalescing for large kernels
    if use_fused_kernel:
        # Grid: each program handles one output position
        num_outputs = in_n * in_c * out_d * out_h * out_w
        grid = (num_outputs,)

        if return_indices:
            max_pool3d_fused_kernel_with_indices[grid](
                input_5d,
                output,
                indices,
                input_5d.stride(0),
                input_5d.stride(1),
                input_5d.stride(2),
                input_5d.stride(3),
                input_5d.stride(4),
                in_n,
                in_c,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                padding_d,
                padding_h,
                padding_w,
                dilation_d,
                dilation_h,
                dilation_w,
            )
        else:
            max_pool3d_fused_kernel[grid](
                input_5d,
                output,
                input_5d.stride(0),
                input_5d.stride(1),
                input_5d.stride(2),
                input_5d.stride(3),
                input_5d.stride(4),
                in_n,
                in_c,
                in_d,
                in_h,
                in_w,
                out_d,
                out_h,
                out_w,
                kernel_d,
                kernel_h,
                kernel_w,
                stride_d,
                stride_h,
                stride_w,
                padding_d,
                padding_h,
                padding_w,
                dilation_d,
                dilation_h,
                dilation_w,
            )

        # Squeeze output back to 4D if input was 4D
        if was_squeezed:
            output = output.squeeze(0)
            if return_indices:
                indices = indices.squeeze(0)
                return output, indices
            return output

        if return_indices:
            return output, indices
        return output

    # Launch kernel
    # Use standard kernels with autotune for all kernel sizes (following max_pool2d pattern)
    grid = lambda meta: (
        in_n * in_c,
        triton.cdiv(out_d, meta["BLOCK_D"])
        * triton.cdiv(out_h, meta["BLOCK_H"])
        * triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    if return_indices:
        max_pool3d_forward_kernel[grid](
            input_5d,
            output,
            indices,
            input_5d.stride(0),
            input_5d.stride(1),
            input_5d.stride(2),
            input_5d.stride(3),
            input_5d.stride(4),
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
        )
    else:
        max_pool3d_forward_kernel_no_indices[grid](
            input_5d,
            output,
            input_5d.stride(0),
            input_5d.stride(1),
            input_5d.stride(2),
            input_5d.stride(3),
            input_5d.stride(4),
            in_c,
            in_d,
            in_h,
            in_w,
            out_d,
            out_h,
            out_w,
            kernel_d,
            kernel_h,
            kernel_w,
            stride_d,
            stride_h,
            stride_w,
            padding_d,
            padding_h,
            padding_w,
            dilation_d,
            dilation_h,
            dilation_w,
        )

    # Squeeze output back to 4D if input was 4D
    if was_squeezed:
        output = output.squeeze(0)
        if return_indices:
            indices = indices.squeeze(0)
            return output, indices
        return output

    if return_indices:
        return output, indices
    return output
