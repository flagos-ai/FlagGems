import logging

import torch
import triton
import triton.language as tl

import flag_gems.runtime as runtime
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device


@triton.heuristics(runtime.get_heuristic_config("zeros"))
@libentry()
@triton.jit
def zeros_kernel(
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(output_ptr + offsets, 0.0, mask=mask)


def zeros(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("METAX GEMS ZEROS")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        zeros_kernel[grid_fn](out, N)
    return out
