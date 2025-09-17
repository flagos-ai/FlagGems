import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(configs=runtime.get_tuned_config("polar"), key=["n_input"])
@triton.jit
def polar_kernel_kernel(
    abs,
    angle,
    output,
    n_input: tl.constexpr,
    n_output: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_input

    inp_abs = tl.load(abs + offset, mask=mask)
    inp_angle = tl.load(angle + offset, mask=mask)
    out_abs = inp_abs * tl.cos(inp_angle)
    out_angle = inp_abs * tl.sin(inp_angle)

    # interleave abs and angle for complex type results
    results = tl.interleave(out_abs, out_angle)
    output_offset = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    output_mask = output_offset < n_output
    tl.store(output + output_offset, results, mask=output_mask)


def polar(abs, angle):
    logger.debug("METAX GEMS polar")
    output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)
    n_input = abs.numel()
    n_output = output.numel()

    grid = lambda meta: (triton.cdiv(n_output, meta["BLOCK_SIZE"]),)
    polar_kernel_kernel[grid](abs, angle, output, n_input, n_output)

    return torch.view_as_complex(output)
