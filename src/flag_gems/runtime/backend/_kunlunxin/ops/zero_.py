import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zero_(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ZERO_")
    N = x.numel()
    if N == 0:
        return x
    grid_fn = (12, 1, 1)
    block_size = triton.next_power_of_2(triton.cdiv(N, 12))
    with torch_device_fn.device(x.device):
        zeros_kernel[grid_fn](
            x,
            N,
            BLOCK_SIZE=block_size,
            buffer_size_limit=4096,
            isCloseDtypeConvert=True,
        )
    return x
