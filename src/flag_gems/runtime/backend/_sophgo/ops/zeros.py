import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

device_ = device


@libentry()
@triton.jit
def zeros_fill_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    zeros = tl.zeros([BLOCK_SIZE], dtype=output_ptr.type.element_ty)
    tl.store(output_ptr + offsets, zeros, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS ZEROS SOPHGO")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)
    out = torch.empty(size, device=device, dtype=dtype)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size = 4096
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(out.device):
        zeros_fill_kernel[grid](
            out,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    return out
