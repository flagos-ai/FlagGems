import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["N"],
)
@triton.jit
def pixel_shuffle_kernel(
    inp_ptr,
    out_ptr,
    N,
    C_out,
    H_out,
    W_out,
    C_in,
    H_in,
    W_in,
    r,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Decode flat output index -> (n, c_out, h_out, w_out)
    tmp = offsets
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c_out = tmp % C_out
    n = tmp // C_out

    # Compute input indices
    dx = w_out % r
    dy = h_out % r
    w_in = w_out // r
    h_in = h_out // r
    c_in = c_out * (r * r) + dy * r + dx

    # Flat input index
    inp_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in

    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def pixel_shuffle(inp, upscale_factor):
    logger.debug("GEMS PIXEL SHUFFLE")
    assert inp.dim() == 4, "pixel_shuffle expects 4D input"
    N, C_in, H_in, W_in = inp.shape
    r = upscale_factor
    if N * C_in * H_in * W_in < 2097152:
        C_out = C_in // (r * r)
        H_out = H_in * r
        W_out = W_in * r
        x = inp.reshape(N, C_out, r, r, H_in, W_in)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(N, C_out, H_out, W_out).contiguous()
    assert C_in % (r * r) == 0, "C_in must be divisible by upscale_factor^2"

    C_out = C_in // (r * r)
    H_out = H_in * r
    W_out = W_in * r

    inp = inp.contiguous()
    out = torch.empty((N, C_out, H_out, W_out), dtype=inp.dtype, device=inp.device)

    total = N * C_out * H_out * W_out
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(inp.device):
        pixel_shuffle_kernel[grid](
            inp, out, total, C_out, H_out, W_out, C_in, H_in, W_in, r
        )

    return out