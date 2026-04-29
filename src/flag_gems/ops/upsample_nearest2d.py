import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _upsample_nearest2d_kernel(
    in_ptr,
    out_ptr,
    batch,
    channels,
    in_h,
    in_w,
    out_h,
    out_w,
    h_scale,  # in_h / out_h  (float, passed as constexpr-friendly scalar)
    w_scale,
    BLOCK: tl.constexpr,
):
    """Nearest-neighbour 2-D upsample kernel."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = batch * channels * out_h * out_w
    mask = offsets < numel

    # Decompose flat index -> (n, c, oh, ow)
    ow = offsets % out_w
    tmp = offsets // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    c = tmp % channels
    n = tmp // channels

    # Nearest-neighbour source coordinates
    ih = tl.minimum((oh * h_scale).to(tl.int32), in_h - 1)
    iw = tl.minimum((ow * w_scale).to(tl.int32), in_w - 1)

    in_idx = n * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def upsample_nearest2d(
    input: torch.Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    """Nearest-neighbour 2-D upsample.

    Args:
        input: 4-D tensor (N, C, H, W).
        output_size: [out_H, out_W].
        scales_h: Optional explicit height scale factor.
        scales_w: Optional explicit width scale factor.

    Returns:
        Upsampled tensor of shape (N, C, out_H, out_W).
    """
    logger.debug("GEMS UPSAMPLE_NEAREST2D")
    assert input.ndim == 4, "upsample_nearest2d expects 4-D input (N, C, H, W)"
    n, c, in_h, in_w = input.shape
    out_h, out_w = output_size

    h_scale = scales_h if scales_h is not None else in_h / out_h
    w_scale = scales_w if scales_w is not None else in_w / out_w

    inp = input.contiguous()
    out = torch.empty((n, c, out_h, out_w), dtype=input.dtype, device=input.device)

    numel = n * c * out_h * out_w
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(input.device):
        _upsample_nearest2d_kernel[grid](
            inp,
            out,
            n,
            c,
            in_h,
            in_w,
            out_h,
            out_w,
            h_scale,
            w_scale,
            BLOCK=BLOCK,
        )
    return out
