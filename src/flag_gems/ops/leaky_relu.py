import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def leaky_relu_kernel(
    X,
    OUT,
    negative_slope,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=0.0)
    out = tl.where(x >= 0, x, negative_slope * x)

    tl.store(OUT + offsets, out, mask=mask)


def leaky_relu(input, negative_slope=0.01):
    logger.debug("GEMS LEAKY_RELU")

    out = torch.empty_like(input)
    n_elements = input.numel()

    if n_elements == 0:
        return out

    # Use larger block size for better performance
    BLOCK = 2048
    grid = (triton.cdiv(n_elements, BLOCK),)
    leaky_relu_kernel[grid](
        input.reshape(-1),
        out.reshape(-1),
        negative_slope,
        n_elements,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out
