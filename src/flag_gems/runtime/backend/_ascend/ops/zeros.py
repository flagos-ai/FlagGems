import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device
logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

MAX_GRID = 65535
BLOCK_SIZE_SUB = 1024


def _compute_block_size(N):
    """Compute BLOCK_SIZE ensuring grid dim (coreDim) <= MAX_GRID.
    BLOCK_SIZE is always a multiple of BLOCK_SIZE_SUB."""
    block_size = BLOCK_SIZE_SUB
    while triton.cdiv(N, block_size) > MAX_GRID:
        block_size *= 2
    return max(block_size, 20480)


@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tle.program_id(axis=0)

    for sub_block_start_idx in range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        sub_offset = (
            pid * BLOCK_SIZE + sub_block_start_idx + tl.arange(0, BLOCK_SIZE_SUB)
        )
        mask = sub_offset < n_elements
        tl.store(output_ptr + sub_offset, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ASCEND ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    if N == 0:
        return out
    BLOCK_SIZE = _compute_block_size(N)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_SUB=BLOCK_SIZE_SUB)
    return out


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_ASCEND ZERO_")
    N = x.numel()
    if N == 0:
        return x
    BLOCK_SIZE = _compute_block_size(N)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](x, N, BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_SUB=BLOCK_SIZE_SUB)
    return x
