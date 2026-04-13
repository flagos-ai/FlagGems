import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def linspace_kernel(
    out_ptr,
    start,
    end,
    step_size,
    steps,
    last_idx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps
    # Compute value from start
    out_val = start + step_size * idx
    # Override last element to be exactly end
    out_val = tl.where(idx == last_idx, end, out_val)
    tl.store(out_ptr + idx, out_val, mask=mask)


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS LINSPACE")
    assert steps >= 1, "steps must be >= 1"

    out = torch.empty(
        steps,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    if steps == 1:
        return torch.fill(out, start)
    else:
        if isinstance(start, torch.Tensor):
            start = start.item()
        if isinstance(end, torch.Tensor):
            end = end.item()
        step_size = (float(end) - float(start)) / (steps - 1)
        last_idx = steps - 1
        # Tuned BLOCK_SIZE and num_warps for memory throughput
        if steps >= 1000000:
            BLOCK_SIZE = 16384
            num_warps = 16
        elif steps >= 100000:
            BLOCK_SIZE = 8192
            num_warps = 8
        elif steps >= 10000:
            BLOCK_SIZE = 4096
            num_warps = 8
        else:
            BLOCK_SIZE = 2048
            num_warps = 4
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        linspace_kernel[grid](
            out,
            start,
            end,
            step_size,
            steps,
            last_idx,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out
