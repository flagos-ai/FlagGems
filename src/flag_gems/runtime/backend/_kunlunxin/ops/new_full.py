import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["fill_value"])
def new_full_kernel(output_ptr, n_elements, fill_value, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + offsets, fill_value, mask=offsets < n_elements)


def _launch_config(n_elements):
    grid = 12
    block_size = triton.next_power_of_2(triton.cdiv(n_elements, grid))
    return (grid, 1, 1), block_size


def new_full(
    self,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_KUNLUNXIN NEW_FULL")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype

    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid, block_size = _launch_config(n_elements)
    with torch_device_fn.device(device):
        new_full_kernel[grid](
            out,
            n_elements,
            fill_value,
            BLOCK_SIZE=block_size,
            buffer_size_limit=2048,
            isCloseDtypeConvert=True,
        )
    return out
