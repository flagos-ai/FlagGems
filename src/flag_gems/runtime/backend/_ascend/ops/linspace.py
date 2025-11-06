import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


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
    pnum = tle.num_programs(0)
    work_loads = tl.cdiv(steps, BLOCK_SIZE)
    loop_counts = tl.cdiv(work_loads, pnum)
    for loop in range(0, loop_counts):
        idx = (pid * loop_counts + loop) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < steps
        fw_mask = idx < mid
        fw_values = start + (step_size * idx)
        bd_values = end - step_size * (steps - idx - 1)

        out_val = tl.where(fw_mask, fw_values, bd_values)
        tl.store(out_ptr + idx * out_stride0, out_val, mask=mask)


def linspace(
    start, end, steps, *, dtype=None, layout=None, device=None, pin_memory=None
) -> torch.Tensor:
    logger.debug("GEMS_ASCEND LINSPACE")
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

        def grid(meta):
            dim0 = triton.cdiv(steps, BLOCK_SIZE)
            while dim0 >= 65536:
                dim0 = triton.cdiv(dim0, 2)
            return (dim0,)

        linspace_kernel[grid](
            out, out.stride(0), start, mid, end, step_size, steps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
