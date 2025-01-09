import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.shape_utils import volume

device_ = device


def heur_block(args):
    if args["N"] <= 1024:
        return 1024
    elif args["N"] <= 2048:
        return 2048
    else:
        return 4096


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 4
    elif args["N"] <= 2048:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "BLOCK_SIZE": heur_block,
        "num_warps": heur_num_warps,
    }
)
@libentry()
@triton.jit
def ones_kernel(
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(output_ptr + offsets, 1.0, mask=mask)


def ones(size, *, dtype=None, layout=None, device=None, pin_memory=None):
    logging.debug("GEMS ONES")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device(device_.name)

    out = torch.empty(size, device=device, dtype=dtype)
    N = volume(size)
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        ones_kernel[grid_fn](out, N)
    return out
