import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logaddexp_kernel(
    X,
    Y,
    OUT,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask).to(tl.float32)
    y = tl.load(Y + offsets, mask=mask).to(tl.float32)

    m = tl.maximum(x, y)
    out = m + tl.log(tl.exp(x - m) + tl.exp(y - m))

    tl.store(OUT + offsets, out, mask=mask)


def logaddexp(X, Y):
    logger.debug("GEMS LOGADDEXP")

    out = torch.empty_like(X, dtype=torch.promote_types(X.dtype, Y.dtype))
    n_elements = X.numel()

    if n_elements == 0:
        return out

    BLOCK = 2048
    grid = (triton.cdiv(n_elements, BLOCK),)
    logaddexp_kernel[grid](
        X.reshape(-1),
        Y.reshape(-1),
        out.reshape(-1),
        n_elements,
        BLOCK=BLOCK,
    )
    return out
