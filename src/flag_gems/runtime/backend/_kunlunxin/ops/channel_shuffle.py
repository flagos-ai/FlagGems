import logging

import torch
import triton

from flag_gems.ops.channel_shuffle import channel_shuffle_kernel
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger("flag_gems.ops.channel_shuffle")


def channel_shuffle(input: torch.Tensor, groups: int) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN CHANNEL_SHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    if x.ndim < 3:
        raise ValueError(
            f"Input must have at least 3 dimensions (C, H, W), got {x.ndim}"
        )

    N, C, H, W = x.shape[-4:]
    g = int(groups)
    assert g > 0, "groups must be > 0"
    assert C % g == 0, f"C ({C}) must be divisible by groups ({g})"

    out = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        channel_shuffle_kernel[grid](
            x,
            out,
            n_elements,
            N,
            C,
            H,
            W,
            g,
            C,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return out
