"""Pixel Shuffle operator implementation."""
import logging

import torch

logger = logging.getLogger(__name__)


def pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Rearranges elements in a tensor from (*, C*r^2, H, W) to (*, C, H*r, W*r).

    This operation is commonly used in super-resolution networks to upscale
    spatial dimensions by rearranging channel data.

    Args:
        input: Input tensor of shape (*, C*r^2, H, W) where * is batch dimensions
        upscale_factor: Factor to increase spatial resolution by (r)

    Returns:
        Output tensor of shape (*, C, H*r, W*r)

    Example:
        >>> input = torch.randn(1, 9, 4, 4)  # C*r^2=9, r=3, so C=1
        >>> output = pixel_shuffle(input, 3)
        >>> output.shape
        torch.Size([1, 1, 12, 12])
    """
    logger.debug("GEMS PIXEL_SHUFFLE")

    # Validate input
    if input.ndim < 3:
        raise ValueError(
            f"pixel_shuffle expects input with at least 3 dimensions, "
            f"but got input with {input.ndim} dimensions"
        )

    if upscale_factor <= 0:
        raise ValueError(f"upscale_factor must be positive, but got {upscale_factor}")

    # Get input shape
    # input shape: (*, C*r^2, H, W)
    batch_dims = input.shape[:-3]  # All dimensions before C, H, W
    channels_in = input.shape[-3]  # C * r^2
    height_in = input.shape[-2]  # H
    width_in = input.shape[-1]  # W

    r = upscale_factor
    r_squared = r * r

    # Validate that channels can be divided by r^2
    if channels_in % r_squared != 0:
        raise ValueError(
            f"pixel_shuffle expects input channels to be divisible by "
            f"upscale_factor^2 ({r_squared}), but got {channels_in} channels"
        )

    channels_out = channels_in // r_squared  # C
    height_out = height_in * r  # H * r
    width_out = width_in * r  # W * r

    # Reshape and permute to rearrange the data
    # Step 1: Reshape (*, C*r^2, H, W) -> (*, C, r, r, H, W)
    input_reshaped = input.reshape(*batch_dims, channels_out, r, r, height_in, width_in)

    # Step 2: Permute to (*, C, H, r, W, r)
    # We need to interleave the r dimensions with H and W
    ndim = len(batch_dims)
    # Permutation: (batch_dims..., C, r, r, H, W) -> (batch_dims..., C, H, r, W, r)
    perm = list(range(ndim))  # batch dimensions stay in place
    perm.extend([ndim, ndim + 3, ndim + 1, ndim + 4, ndim + 2])
    # ndim: C, ndim+1: r (height), ndim+2: r (width), ndim+3: H, ndim+4: W
    # Result: (batch..., C, H, r_h, W, r_w)

    input_permuted = input_reshaped.permute(*perm)

    # Step 3: Reshape to (*, C, H*r, W*r)
    output = input_permuted.reshape(*batch_dims, channels_out, height_out, width_out)

    return output
