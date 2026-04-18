import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    batch,
    c_out,
    h_out,
    w_out,
    r,          # upscale_factor
    BLOCK: tl.constexpr,
):
    """Pixel shuffle: (N, C*r^2, H, W) -> (N, C, H*r, W*r)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    numel = batch * c_out * h_out * w_out
    mask = offsets < numel

    # Decompose output flat index
    ow = offsets % w_out
    tmp = offsets // w_out
    oh = tmp % h_out
    tmp = tmp // h_out
    oc = tmp % c_out
    n = tmp // c_out

    # Map back to input indices
    # oh = ih * r + rh,  ow = iw * r + rw
    rh = oh % r
    ih = oh // r
    rw = ow % r
    iw = ow // r

    # Input channel: ic = oc * r^2 + rh * r + rw
    ic = oc * r * r + rh * r + rw
    c_in = c_out * r * r
    h_in = h_out // r
    w_in = w_out // r

    in_idx = n * (c_in * h_in * w_in) + ic * (h_in * w_in) + ih * w_in + iw
    val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """Rearrange elements in a tensor of shape (N, C*r^2, H, W) to (N, C, H*r, W*r).

    Args:
        input: 4-D tensor (N, C*r^2, H, W).
        upscale_factor: r, the upscale factor.

    Returns:
        Tensor of shape (N, C, H*r, W*r).
    """
    logger.debug("GEMS PIXEL_SHUFFLE")
    assert input.ndim == 4, "pixel_shuffle expects 4-D input"
    n, c_in, h_in, w_in = input.shape
    r = upscale_factor
    assert c_in % (r * r) == 0, f"Input channels ({c_in}) must be divisible by r^2 ({r*r})"

    c_out = c_in // (r * r)
    h_out = h_in * r
    w_out = w_in * r

    inp = input.contiguous()
    out = torch.empty((n, c_out, h_out, w_out), dtype=input.dtype, device=input.device)

    numel = n * c_out * h_out * w_out
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)

    with torch_device_fn.device(input.device):
        _pixel_shuffle_kernel[grid](
            inp,
            out,
            n,
            c_out,
            h_out,
            w_out,
            r,
            BLOCK=BLOCK,
        )
    return out
