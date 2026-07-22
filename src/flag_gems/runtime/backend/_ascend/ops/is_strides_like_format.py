# src/flag_gems/runtime/backend/_ascend/ops/is_strides_like_format.py
import logging

import torch

from flag_gems.runtime import device

device_ = device
logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

SUPPORTED_FORMATS = {"channels_last", "channels_last_3d"}


def is_strides_like_format(x: torch.Tensor, format: str) -> bool:
    """
    Check if tensor's stride matches the specified memory format.

    Only supports 'channels_last' (4D) and 'channels_last_3d' (5D).
    All other formats return False immediately.

    Args:
        x: Input tensor.
        format: Target memory format. Supported values:
                "channels_last" - 4D NHWC stride pattern (N, C, H, W)
                "channels_last_3d" - 5D NDHWC stride pattern (N, C, D, H, W)

    Returns:
        bool: True if stride matches, False otherwise.
    """
    logger.debug(f"GEMS_ASCEND IS_STRIDES_LIKE_FORMAT: format={format}")

    if format == "channels_last":
        if x.dim() != 4:
            return False
        N, C, H, W = x.shape
        expected = (C * H * W, 1, W * C, C)
        return x.stride() == expected

    if format == "channels_last_3d":
        if x.dim() != 5:
            return False
        N, C, D, H, W = x.shape
        expected = (C * D * H * W, 1, C * W * H, C * W, C)
        return x.stride() == expected

    # Unsupported format
    return False
