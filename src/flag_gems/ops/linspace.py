import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linspace_kernel(
    out_ptr,
    out_stride0,
    start,
    mid,
    end,
    step_size,
    steps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < steps
    fw_mask = idx < mid
    fw_values = start + (step_size * idx)
    bd_values = end - step_size * (steps - idx - 1)

    out_val = tl.where(fw_mask, fw_values, bd_values)
    tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)


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
        mid = steps // 2
        step_size = (float(end) - float(start)) / (steps - 1)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(steps, BLOCK_SIZE),)
        linspace_kernel[grid](
            out, out.stride(0), start, mid, end, step_size, steps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
