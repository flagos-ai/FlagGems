import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)
INT32_MAX = 2**31 - 1

AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=4),
]


@libentry()
@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["hw_elements", "H_out", "W_out"])
@triton.heuristics({"USE_INT32_IDX": lambda args: args["n_elements"] <= INT32_MAX})
@triton.jit
def pixel_shuffle_kernel(
    in_ptr,
    out_ptr,
    H,
    H_out,
    W,
    W_out,
    hw_elements,
    n_elements,
    R: tl.constexpr,
    R2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT32_IDX: tl.constexpr,
):
    if USE_INT32_IDX:
        pid = tle.program_id(0)
        batch_id = tle.program_id(1)
    else:
        pid = tle.program_id(0).to(tl.int64)
        batch_id = tle.program_id(1).to(tl.int64)

    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < hw_elements

    wo = offs % W_out
    tmp = offs // W_out
    ho = tmp % H_out
    co = tmp // H_out

    rh = ho % R
    h = ho // R
    rw = wo % R
    w = wo // R
    cin = co * R2 + rh * R + rw

    batch_offset = batch_id * hw_elements
    in_idx = batch_offset + cin * H * W + h * W + w
    out_idx = batch_offset + offs

    val = tl.load(in_ptr + in_idx, mask=mask, other=0)
    tl.store(out_ptr + out_idx, val, mask=mask)


def pixel_shuffle(self: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    logger.debug("GEMS PIXEL_SHUFFLE")
    assert self.ndim >= 3, (
        f"pixel_shuffle expects input to have at least 3 dimensions, "
        f"but got input with {self.ndim} dimension(s)"
    )
    r = upscale_factor
    assert r > 0, f"pixel_shuffle expects upscale_factor > 0, but got {r}"
    r2 = r * r
    C_in = self.shape[-3]
    H = self.shape[-2]
    W = self.shape[-1]
    assert C_in % r2 == 0, (
        f"pixel_shuffle expects input channel to be divisible by "
        f"upscale_factor^2={r2}, but got {C_in} channels"
    )

    C_out = C_in // r2
    H_out = H * r
    W_out = W * r

    batch_shape = self.shape[:-3]
    batch = 1
    for s in batch_shape:
        batch *= s

    out_shape = batch_shape + (C_out, H_out, W_out)
    output = torch.empty(out_shape, dtype=self.dtype, device=self.device)

    if self.numel() == 0:
        return output

    inp = self if self.is_contiguous() else self.contiguous()
    inp_view = inp.reshape(batch, C_in, H, W)
    out_view = output.reshape(batch, C_out, H_out, W_out)
    hw_elements = C_out * H_out * W_out
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(hw_elements, meta["BLOCK_SIZE"]), batch)

    with torch_device_fn.device(self.device):
        pixel_shuffle_kernel[grid](
            inp_view,
            out_view,
            H,
            H_out,
            W,
            W_out,
            hw_elements,
            n_elements,
            R=r,
            R2=r2,
        )

    return output
