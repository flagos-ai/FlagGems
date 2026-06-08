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
def ceil_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.ceil(x)
        tl.store(out_ptr + off, out, mask=mask)


def _choose_params(N_total):
    if N_total <= 65536:
        block = triton.next_power_of_2(N_total)
        if block < 1024:
            block = 1024
        return block, 2
    return 65536, 2 if N_total <= 134217728 else 4


def _launch_ceil(inp, out, N_total):
    BLOCK, nw = _choose_params(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        ceil_flat_kernel[(grid_size,)](
            inp,
            out,
            N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )


def ceil(A):
    logger.debug("GEMS CEIL")
    inp = A.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)
    _launch_ceil(inp, out, N_total)
    return out


def ceil_(A):
    logger.debug("GEMS CEIL_")
    inp = A.contiguous()
    N_total = inp.numel()
    _launch_ceil(inp, A, N_total)
    return A
