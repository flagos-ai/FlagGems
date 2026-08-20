import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils.shape_utils import volume

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)
device_ = device


@triton.jit
def empty_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, 0.0, mask=mask)


def empty(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Returns a tensor filled with uninitialized data."""
    logger.debug("GEMS_CAMBRICON EMPTY")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if memory_format is None:
        memory_format = torch.contiguous_format

    # Allocate via empty_strided instead of torch.empty to avoid self-recursion
    # through aten::empty.memory_format.
    shape = tuple(size)
    meta = torch.empty(shape, dtype=dtype, device="meta", memory_format=memory_format)
    out = torch.empty_strided(
        shape,
        meta.stride(),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
    N = volume(shape)
    grid_fn = lambda meta: (min(triton.cdiv(N, meta["BLOCK_SIZE"]), TOTAL_CORE_NUM),)
    with torch_device_fn.device(device):
        empty_kernel[grid_fn](out, N, BLOCK_SIZE=1024)
    return out
