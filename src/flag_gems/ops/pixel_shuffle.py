import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    C_out,
    H,
    W,
    R,
    H_out,
    W_out,
    s_in_n,
    s_in_c,
    s_in_h,
    s_in_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decompose linear index into (n, c_out, h_out, w_out)
    wo = offs % W_out
    tmp = offs // W_out
    ho = tmp % H_out
    tmp = tmp // H_out
    co = tmp % C_out
    no = tmp // C_out

    # Map output coords back to input coords
    rh = ho % R
    h = ho // R
    rw = wo % R
    w = wo // R

    cin = co * (R * R) + rh * R + rw

    in_idx = no * s_in_n + cin * s_in_c + h * s_in_h + w * s_in_w

    # Output is contiguous (N, C_out, H_out, W_out)
    val = tl.load(in_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + offs, val, mask=mask)


def pixel_shuffle(self, upscale_factor):
    logger.debug("GEMS PIXEL_SHUFFLE")

    assert self.dim() == 4, (
        f"pixel_shuffle expects a 4D tensor (N, C, H, W), but got {self.dim()}D"
    )
    assert isinstance(upscale_factor, int) and upscale_factor > 0, (
        "upscale_factor must be a positive integer"
    )

    N, C_in, H, W = self.shape
    r2 = upscale_factor * upscale_factor
    assert C_in % r2 == 0, (
        f"Input channel dimension {C_in} is not divisible by upscale_factor^2={r2}"
    )

    C_out = C_in // r2
    H_out = H * upscale_factor
    W_out = W * upscale_factor

    out = torch.empty(
        (N, C_out, H_out, W_out),
        dtype=self.dtype,
        device=self.device,
    )

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    with torch_device_fn.device(self.device):
        pixel_shuffle_kernel[grid](
            self,
            out,
            C_out,
            H,
            W,
            upscale_factor,
            H_out,
            W_out,
            self.stride(0),
            self.stride(1),
            self.stride(2),
            self.stride(3),
            n_elements,
            BLOCK_SIZE,
        )

    return out
