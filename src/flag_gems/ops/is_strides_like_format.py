# src/flag_gems/ops/is_strides_like_format.py
import torch

SUPPORTED_FORMATS = {"channels_last", "channels_last_3d"}


def is_strides_like_format(x: torch.Tensor, format: str) -> bool:
    """Check if tensor's stride matches the specified memory format.

    Only supports 'channels_last' (4D) and 'channels_last_3d' (5D).
    All other formats return False immediately.
    """
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
