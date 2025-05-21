import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


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
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    value_scalar = tl.load(value_ptr)  # load the value from the tensor.
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    logger.debug("METAX GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out


def fill_scalar(input, value):
    logger.debug("METAX GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid_fn](out, N, value)
    return out
