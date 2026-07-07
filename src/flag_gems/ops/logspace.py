import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def logspace_kernel(
    out_ptr,
    out_stride0,
    start,
    step_size,
    steps,
    log2_base: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps

    exponent = start + idx * step_size
    vals = tl.exp2(log2_base * exponent)

    tl.store(out_ptr + idx * out_stride0, vals, mask=mask)


def logspace(
    start,
    end,
    steps,
    base=10.0,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> torch.Tensor:
    logger.debug("GEMS LOGSPACE")
    assert steps >= 0, "number of steps must be non-negative"
    out_dtype = dtype if dtype is not None else torch.get_default_dtype()
    out = torch.empty(
        steps,
        dtype=out_dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 0:
        pass
    elif steps == 1:
        if isinstance(start, torch.Tensor):
            start = start.item()
        out = torch.fill(out, base**start)
    else:
        if isinstance(start, torch.Tensor):
            start = start.item()
        if isinstance(end, torch.Tensor):
            end = end.item()
        step_size = (float(end) - float(start)) / (steps - 1)
        BLOCK_SIZE = 256  # according to benchmark, 256 is the best block size
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        logspace_kernel[grid](
            out,
            out.stride(0),
            start,
            step_size,
            steps,
            log2_base=math.log2(float(base)),  # math.log2 require float input
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return out
