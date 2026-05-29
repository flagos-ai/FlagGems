import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import tl_extra_shim

_acos = tl_extra_shim.acos

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N"])
def acos_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = _acos(x)
        tl.store(out_ptr + off, out, mask=mask)


def _choose_block(N, is_fp32):
    if N <= 1024:
        return 1024
    if N <= 32768:
        return triton.next_power_of_2(N)
    if is_fp32:
        return 65536
    return 4096


def acos(A):
    logger.debug("GEMS ACOS GCU400")
    inp = A if A.is_contiguous() else A.contiguous()
    N = inp.numel()
    out = torch.empty_like(inp)

    is_fp32 = inp.dtype == torch.float32
    BLOCK = _choose_block(N, is_fp32)
    grid_size = min(triton.cdiv(N, BLOCK), NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        acos_kernel[(grid_size,)](
            inp, out, N,
            BLOCK=BLOCK, num_warps=4,
        )
    return out
