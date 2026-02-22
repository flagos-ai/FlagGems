import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["N", "C_out", "H_out", "W_out"],
)
@triton.jit
def pixel_shuffle_kernel(
    inp_ptr,
    out_ptr,
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    r,
    USE_INT32_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Map output (n, c_out, h_out, w_out) <- input (n, c_in, h_in, w_in).

    Given upscale_factor r:
        dx    = w_out % r
        dy    = h_out % r
        w_in  = w_out // r
        h_in  = h_out // r
        c_in  = c_out * r * r + dy * r + dx
    """
    if USE_INT32_IDX:
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = N * C_out * H_out * W_out
    else:
        pid = tl.program_id(0).to(tl.int64)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        total = N.to(tl.int64) * C_out * H_out * W_out

    mask = idx < total

    # Decompose flat output index -> (n, c_out, h_out, w_out)
    tmp = idx
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    c_out = tmp % C_out
    n = tmp // C_out

    # Compute corresponding input coordinates
    dx = w_out % r
    dy = h_out % r
    w_in = w_out // r
    h_in = h_out // r
    c_in = c_out * r * r + dy * r + dx

    # Flat input index (contiguous layout)
    inp_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in

    val = tl.load(inp_ptr + inp_idx, mask=mask)
    tl.store(out_ptr + idx, val, mask=mask)


def pixel_shuffle(input: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """Rearrange elements from (*, C*r^2, H, W) to (*, C, H*r, W*r).

    Args:
        input: tensor with shape (*, C*r^2, H, W), at least 3-D.
        upscale_factor: spatial upscaling factor r (positive integer).

    Returns:
        tensor with shape (*, C, H*r, W*r).
    """
    logger.debug("GEMS PIXEL_SHUFFLE")

    if input.ndim < 3:
        raise ValueError(
            f"pixel_shuffle expects input with at least 3 dimensions, got {input.ndim}"
        )
    if upscale_factor <= 0:
        raise ValueError(
            f"upscale_factor must be a positive integer, got {upscale_factor}"
        )

    r = upscale_factor
    r2 = r * r

    *batch, C_in, H_in, W_in = input.shape
    if C_in % r2 != 0:
        raise ValueError(
            f"pixel_shuffle expects the number of channels ({C_in}) to be "
            f"divisible by upscale_factor^2 ({r2})"
        )

    C_out = C_in // r2
    H_out = H_in * r
    W_out = W_in * r
    N = 1
    for d in batch:
        N *= d

    # Kernel requires contiguous input
    inp = input.contiguous()
    out = torch.empty(
        (*batch, C_out, H_out, W_out), dtype=input.dtype, device=input.device
    )

    total = N * C_out * H_out * W_out
    USE_INT32_IDX = total <= (2**31 - 1)

    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)  # noqa: E731
    with torch_device_fn.device(input.device):
        pixel_shuffle_kernel[grid](
            inp,
            out,
            N,
            C_in,
            H_in,
            W_in,
            C_out,
            H_out,
            W_out,
            r,
            USE_INT32_IDX=USE_INT32_IDX,
        )
    return out
