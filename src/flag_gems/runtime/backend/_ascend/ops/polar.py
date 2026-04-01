import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

NUM_VECTOR_CORES = 48


@libentry()
@triton.jit
def polar_kernel(
    abs_ptr,
    angle_ptr,
    out_ptr,
    N,
    num_tasks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
    NCORE: tl.constexpr,
):
    pid = tl.program_id(0)

    for task_id in range(pid, num_tasks, NCORE):
        base_offset = task_id * BLOCK_SIZE

        for sub_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = base_offset + sub_idx + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N

            abs_val = tl.load(abs_ptr + offsets, mask=mask, care_padding=False)
            angle_val = tl.load(angle_ptr + offsets, mask=mask, care_padding=False)

            real = abs_val * tl.cos(angle_val)
            imag = abs_val * tl.sin(angle_val)

            # Interleave real and imag for complex output layout
            results = tl.interleave(real, imag)
            out_offsets = (base_offset + sub_idx) * 2 + tl.arange(0, BLOCK_SIZE_SUB * 2)
            out_mask = out_offsets < N * 2
            tl.store(out_ptr + out_offsets, results, mask=out_mask)


def polar(abs, angle):
    logger.debug("GEMS_ASCEND POLAR")
    # view_as_complex does not support bfloat16, cast to float32 for computation
    input_dtype = abs.dtype
    if input_dtype == torch.bfloat16:
        abs = abs.to(torch.float32)
        angle = angle.to(torch.float32)

    abs = abs.contiguous()
    angle = angle.contiguous()
    N = abs.numel()

    output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)

    BLOCK_SIZE = 1024
    BLOCK_SIZE_SUB = 1024
    num_tasks = triton.cdiv(N, BLOCK_SIZE)
    ncore = min(num_tasks, NUM_VECTOR_CORES)
    grid = (ncore,)

    polar_kernel[grid](abs, angle, output, N, num_tasks, BLOCK_SIZE, BLOCK_SIZE_SUB, ncore)

    return torch.view_as_complex(output)
