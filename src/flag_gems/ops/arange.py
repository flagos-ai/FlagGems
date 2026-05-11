import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    y_ptr += pid * BLOCK_SIZE
    step_offset = pid * BLOCK_SIZE * step

    cols = tl.arange(0, BLOCK_SIZE)
    arange_val = cols * step + step_offset + start
    mask = cols + pid * BLOCK_SIZE
    tl.store(y_ptr + cols, arange_val, mask=mask < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS ARANGE")
    if dtype is torch.int64:
        start = int(start)
        end = int(end)
        step = int(step)
        if step == 0:
            raise RuntimeError("step must be nonzero")
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        if dtype is torch.int64 and (
            isinstance(step, float)
            or isinstance(start, float)
            or isinstance(end, float)
        ):
            int_step = int(step)
            if int_step == 0:
                raise RuntimeError("step must be nonzero")
        size = math.ceil((end - start) / step)
    size = int(size)

    BLOCK_SIZE = 128
    grid = triton.cdiv(size, BLOCK_SIZE)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = (
            runtime.device.name
        )  # Note(Zhengzekang): Torch default value is CPU, but triton is target to GPU.

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    arange_func[grid,](result, start, end, step, size, BLOCK_SIZE)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
