import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def polar_kernel(
    abs_ptr,
    angle_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    abs_val = tl.load(abs_ptr + offsets, mask=mask).to(tl.float32)
    angle_val = tl.load(angle_ptr + offsets, mask=mask).to(tl.float32)

    real = abs_val * tl.cos(angle_val)
    imag = abs_val * tl.sin(angle_val)

    out_offsets = offsets * 2
    tl.store(out_ptr + out_offsets, real.to(abs_val.dtype), mask=mask)
    tl.store(out_ptr + out_offsets + 1, imag.to(abs_val.dtype), mask=mask)


def polar(abs, angle):
    logger.debug("GEMS_ILUVATAR POLAR")
    assert abs.shape == angle.shape
    assert abs.is_cuda and angle.is_cuda

    N = abs.numel()
    output_flat = torch.empty(N * 2, dtype=abs.dtype, device=abs.device)

    BLOCK_SIZE = 1024
    grid = triton.cdiv(N, BLOCK_SIZE)

    polar_kernel[grid,](abs, angle, output_flat, N, BLOCK_SIZE)

    output = output_flat.reshape(*abs.shape, 2)
    return torch.view_as_complex(output)
