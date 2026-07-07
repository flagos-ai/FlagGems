import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@libtuner(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_stages=3, num_warps=1),
    ],
    key=["size"],
    strategy=["log"],
)
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    block_step = num_jobs * BLOCK_SIZE
    block_start = block_start
    for block_start_offset in range(block_start, size, block_step):
        offset = tl.arange(0, BLOCK_SIZE) + block_start_offset
        arange_val = offset * step + start
        tl.store(y_ptr + offset, arange_val, mask=offset < size)


def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    logger.debug("GEMS_CAMBRICON ARANGE")
    if dtype is torch.int64:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)

    assert (
        size < torch.iinfo(torch.int32).max
    ), f"Size {size} is not less than the maximum int32 value max_int32"

    grid = lambda META: (min(triton.cdiv(size, META["BLOCK_SIZE"]), TOTAL_CORE_NUM),)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    if device is None:
        device = (
            runtime.device.name
        )  # Note(Zhengzekang): Torch default value is CPU, but triton is target to GPU.

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    arange_func[grid](result, start, end, step, size)
    return result


def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )
