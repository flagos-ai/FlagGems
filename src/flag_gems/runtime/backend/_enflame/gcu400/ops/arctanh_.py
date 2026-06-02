import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def arctanh_kernel_(x_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (N_total + BLOCK - 1) // BLOCK, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        val = 0.5 * tl.log((1.0 + x) / (1.0 - x))
        tl.store(x_ptr + off, val, mask=mask)


def arctanh_(*args, **kwargs):
    logger.debug("GEMS ARCTANH_ GCU400")
    x = None
    if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        x = args[0]
    else:
        x = kwargs.get("input", kwargs.get("self", None))
    if not isinstance(x, torch.Tensor):
        raise TypeError("arctanh_ expects a single Tensor argument")
    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    if not x.is_floating_point():
        raise TypeError("arctanh_ only supports floating point tensors")

    N_total = x.numel()
    if N_total == 0:
        return x

    BLOCK = 8192
    grid_size = min((N_total + BLOCK - 1) // BLOCK, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        arctanh_kernel_[(grid_size,)](x, N_total, BLOCK=BLOCK, num_warps=4)
    return x
