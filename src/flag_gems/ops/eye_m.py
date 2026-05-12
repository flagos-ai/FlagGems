import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
device_ = device


@libentry()
@triton.jit
def eye_kernel(
    out_ptr,
    N,
    M,
    BLOCK_i: tl.constexpr,
    BLOCK_j: tl.constexpr,
):
    pid_i = tl.program_id(0)  # block id
    off_i = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)
    mask_i = off_i < N

    pid_j = tl.program_id(1)  # block id
    off_j = pid_j * BLOCK_j + tl.arange(0, BLOCK_j)
    mask_j = off_j < M

    val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
    mask = mask_i[:, None] & mask_j[None, :]
    off_ij = off_i[:, None] * M + off_j[None, :]

    tl.store(out_ptr + off_ij, val, mask=mask)


def eye_m(n, m, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye_m(n, m), using 2D tiles to split the matrix into blocks.
    """
    logger.debug("GEMS EYE_M")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye_m.")

    out = torch.empty(
        (n, m), dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
    )
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(m, BLOCK_SIZE))

    with torch_device_fn.device(device):
        eye_kernel[grid](
            out,
            n,
            m,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )
    return out
