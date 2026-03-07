import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def log10_kernel(
    X,
    OUT,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=1.0)  # Use 1.0 as default to avoid log(0)
    # Convert to float32 for tl.log (doesn't support fp16)
    x_f32 = x.to(tl.float32)
    # log10(x) = log(x) / log(10)
    # Use precomputed constant: 1/log(10) = 0.43429448190325182765
    out_f32 = tl.log(x_f32) * 0.43429448190325182765
    # Convert back to original dtype
    out = out_f32.to(x.dtype)

    tl.store(OUT + offsets, out, mask=mask)


def log10(A):
    logger.debug("GEMS LOG10")

    out = torch.empty_like(A)
    n_elements = A.numel()

    if n_elements == 0:
        return out

    # Use larger block size and more warps for better performance
    BLOCK = 2048
    grid = (triton.cdiv(n_elements, BLOCK),)
    log10_kernel[grid](
        A.reshape(-1),
        out.reshape(-1),
        n_elements,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out
