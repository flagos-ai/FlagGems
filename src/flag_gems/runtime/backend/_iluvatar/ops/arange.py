import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(output_ptr, start, step, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = start + offsets.to(tl.float32) * step
    tl.store(output_ptr + offsets, values, mask=mask)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_ILUVATAR ARANGE")
    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        if step == 0:
            raise RuntimeError("step must be nonzero")
        size = math.ceil((end - start) / step)
    size = int(size)

    if size == 0:
        if dtype is None:
            dtype = torch.int64
        if device is None:
            device = "cuda"
        return torch.empty((0,), device=device, dtype=dtype)

    BLOCK_SIZE = 2048
    grid = triton.cdiv(size, BLOCK_SIZE)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = "cuda"

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    arange_func[grid,](result, start, step, size, BLOCK_SIZE)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
