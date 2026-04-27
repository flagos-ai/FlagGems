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
    MAX_GRID_DIM_x: tl.constexpr,
    MAX_GRID_DIM_y: tl.constexpr,
):
    pid_i = tl.program_id(0)  # block id
    grim_size_x = (N + MAX_GRID_DIM_x - 1) // MAX_GRID_DIM_x
    pid_j = tl.program_id(1)  # block id
    grim_size_y = (M + MAX_GRID_DIM_y - 1) // MAX_GRID_DIM_y
    for i in range(0, grim_size_x, BLOCK_i):
        block_start_x = pid_i * grim_size_x + i * BLOCK_i
        off_i = block_start_x + tl.arange(0, BLOCK_i)
        mask_i = (off_i < N) & (off_i < pid_i * grim_size_x + grim_size_x)
        for j in range(0, grim_size_y, BLOCK_j):
            block_start_y = pid_j * grim_size_y + j * BLOCK_j
            off_j = block_start_y + tl.arange(0, BLOCK_j)
            mask_j = (off_j < M) & (off_j < pid_j * grim_size_y + grim_size_y)

            val = tl.where(off_i[:, None] == off_j[None, :], 1.0, 0.0)
            mask = mask_i[:, None] & mask_j[None, :]
            off_ij = off_i[:, None] * M + off_j[None, :]

            tl.store(out_ptr + off_ij, val, mask=mask)


def eye(size, *, dtype=None, layout=torch.strided, device=None, pin_memory=None):
    """
    Triton-based implementation of torch.eye(n, n), using 2D tiles to split the matrix into blocks.
    """
    logger.debug("GEMS EYE")

    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout != torch.strided:
        raise ValueError("Currently only strided layout is supported for eye.")

    out = torch.empty(
        (size, size), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
    BLOCK_SIZE = 256
    MAX_GRID_DIM_x = 48
    MAX_GRID_DIM_y = 48
    grid = (MAX_GRID_DIM_x, MAX_GRID_DIM_y)

    with torch_device_fn.device(device):
        eye_kernel[grid](
            out,
            size,
            size,
            BLOCK_SIZE,
            BLOCK_SIZE,
            MAX_GRID_DIM_x,
            MAX_GRID_DIM_y,
        )
    return out
