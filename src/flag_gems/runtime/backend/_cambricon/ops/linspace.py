import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("linspace"))
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
    INNER_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    numcycle = (BLOCK_SIZE + INNER_BLOCK_SIZE - 1) // INNER_BLOCK_SIZE
    for innerid in range(0, numcycle):
        inneridx = innerid * INNER_BLOCK_SIZE + tl.arange(0, INNER_BLOCK_SIZE)
        idx = pid * BLOCK_SIZE + inneridx
        mask = (idx < steps) & (inneridx < BLOCK_SIZE)
        fw_mask = idx < mid
        fw_values = start + (step_size * idx)
        bd_values = end - step_size * (steps - idx - 1)

        out_val = tl.where(fw_mask, fw_values, bd_values)
        tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON LINSPACE")
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
        block_size = triton.cdiv(steps, TOTAL_CORE_NUM)
        grid = (TOTAL_CORE_NUM,)
        linspace_kernel[grid](
            out, out.stride(0), start, mid, end, step_size, steps, BLOCK_SIZE=block_size
        )
        return out
