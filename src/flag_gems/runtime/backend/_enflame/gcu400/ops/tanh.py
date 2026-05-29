import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit
def tanh_flat_kernel(
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
        e2x = tl.exp(2.0 * x)
        out = (e2x - 1.0) / (e2x + 1.0)
        tl.store(out_ptr + off, out, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def tanh_backward_kernel(y, dy):
    y = y.to(tl.float32)
    return dy.to(tl.float32) * (1.0 - y * y)


def _choose_block(N_total):
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    return 65536


def tanh(self):
    logger.debug("GEMS TANH FORWARD")
    inp = self.contiguous()
    N_total = inp.numel()
    out = torch.empty_like(inp)

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp, out, N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return out


def tanh_backward(grad_output, output):
    logger.debug("GEMS TANH BACKWARD")
    in_grad = tanh_backward_kernel(output, grad_output)
    return in_grad


def tanh_(A):
    logger.debug("GEMS TANH_ FORWARD")
    inp = A.contiguous()
    N_total = inp.numel()

    BLOCK = _choose_block(N_total)
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    nw = 2 if N_total <= 134217728 else 4

    with torch_device_fn.device(inp.device):
        tanh_flat_kernel[(grid_size,)](
            inp, A, N_total,
            BLOCK=BLOCK,
            num_warps=nw,
        )

    return A
