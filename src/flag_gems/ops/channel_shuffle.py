import logging

import torch

import flag_gems
from flag_gems.ops.pointwise import pointwise_dynamic

logger = logging.getLogger(__name__)


def channel_shuffle(input: torch.Tensor, groups: int):
    logger.debug("GEMS CHANNEL_SHUFFLE")

    if groups <= 0:
        raise ValueError("groups must be positive")

    if input.dim() < 3:
        raise ValueError("input tensor must have at least 3 dimensions")

    C = input.shape[-3]
    if C % groups != 0:
        raise ValueError(
            f"number of channels must be divisible by groups, got C={C}, groups={groups}"
        )

    if C == 0:
        return input

    channels_per_group = C // groups

    def channel_shuffle_func(x):
        # x shape: (..., C, H, W)
        # Reshape to (..., groups, channels_per_group, H, W)
        # Then transpose to (..., channels_per_group, groups, H, W)
        # Finally reshape back to (..., C, H, W)
        B = x.shape[:-3]
        HW = x.shape[-2:]
        new_shape = B + (groups, channels_per_group) + HW
        reshaped = x.reshape(new_shape)
        # Transpose groups and channels_per_group dimensions
        permuted = reshaped.transpose(-3, -2)
        # Reshape back
        final_shape = B + (C,) + HW
        return permuted.reshape(final_shape)

    return pointwise_dynamic(
        (input,),
        channel_shuffle_func,
        out_shape=input.shape,
        out_dtype=input.dtype,
    )
