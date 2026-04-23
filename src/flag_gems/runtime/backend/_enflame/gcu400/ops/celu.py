import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total", "alpha"])
def celu_kernel(x_ptr, out_ptr, N_total, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(x > 0.0, x, alpha * (tl.exp(x / alpha) - 1.0))
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def celu_kernel_alpha1(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(x > 0.0, x, tl.exp(x) - 1.0)
        tl.store(out_ptr + off, out, mask=mask)


def _launch_celu(inp, out, N_total, alpha):
    is_fp32 = inp.dtype == torch.float32
    if alpha == 1.0:
        if N_total <= 65536:
            BLOCK = triton.next_power_of_2(N_total)
            if BLOCK < 1024:
                BLOCK = 1024
        else:
            BLOCK = 65536 if is_fp32 else 131072
        NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
        grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
        with torch_device_fn.device(inp.device):
            celu_kernel_alpha1[(grid_size,)](inp, out, N_total, BLOCK=BLOCK, num_warps=2)
    else:
        if N_total <= 65536:
            BLOCK = triton.next_power_of_2(N_total)
            if BLOCK < 1024:
                BLOCK = 1024
        else:
            BLOCK = 65536
        NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
        grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
        with torch_device_fn.device(inp.device):
            celu_kernel[(grid_size,)](inp, out, N_total, alpha, BLOCK=BLOCK, num_warps=2)


def celu(A, alpha=1.0):
    logger.debug("GEMS CELU")
    inp = A.contiguous()
    out = torch.empty_like(inp)
    _launch_celu(inp, out, inp.numel(), alpha)
    return out


def celu_(A, alpha=1.0):
    logger.debug("GEMS CELU_")
    inp = A.contiguous()
    _launch_celu(inp, inp, inp.numel(), alpha)
    return A
