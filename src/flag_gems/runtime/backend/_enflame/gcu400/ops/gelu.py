import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim
from flag_gems.utils import libentry

erf = tl_extra_shim.erf

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def gelu_fwd_flat_kernel(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        a = 1.59576912 * x * (1.0 + 0.044715 * x * x)
        sig = 1.0 / (1.0 + tl.exp(-a))
        out = x * sig
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def gelu_bwd_tanh_kernel(x_ptr, dy_ptr, dx_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
        x2 = x * x
        b = 1.59576912 * x * (1.0 + 0.044715 * x2)
        sig = 1.0 / (1.0 + tl.exp(-b))
        db_dx = 1.59576912 + 0.2140644486 * x2
        dydx = sig * (1.0 + x * (1.0 - sig) * db_dx)
        dx = dydx * dy
        tl.store(dx_ptr + off, dx, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def gelu_bwd_none_kernel(x_ptr, dy_ptr, dx_ptr, N_total, BLOCK: tl.constexpr):
    SCALE1: tl.constexpr = 0.7071067811
    SCALE2: tl.constexpr = 0.3989422803
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        dy = tl.load(dy_ptr + off, mask=mask).to(tl.float32)
        sx = SCALE1 * x
        dydx = SCALE2 * x * tl.exp(-sx * sx) + 0.5 * erf(sx) + 0.5
        dx = dydx * dy
        tl.store(dx_ptr + off, dx, mask=mask)


def _launch_fwd(inp, out, N_total):
    is_fp32 = inp.dtype == torch.float32
    if N_total <= 1024:
        BLOCK = 1024
    elif N_total <= 65536:
        BLOCK = triton.next_power_of_2(N_total)
    else:
        BLOCK = 65536 if is_fp32 else 131072
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    with torch_device_fn.device(inp.device):
        gelu_fwd_flat_kernel[(grid_size,)](inp, out, N_total, BLOCK=BLOCK, num_warps=4)


def _launch_bwd(x, dy, dx, N_total, approximate="none"):
    BLOCK = 2048
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)
    kernel = gelu_bwd_tanh_kernel if approximate == "tanh" else gelu_bwd_none_kernel
    with torch_device_fn.device(x.device):
        kernel[(grid_size,)](x, dy, dx, N_total, BLOCK=BLOCK, num_warps=4)


def gelu(self, *, approximate="none"):
    logger.debug("GEMS GELU FORWARD")
    inp = self.contiguous()
    out = torch.empty_like(inp)
    N_total = inp.numel()
    _launch_fwd(inp, out, N_total)
    return out


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS GELU BACKWARD")
    x = self.contiguous()
    dy = grad_output.contiguous()
    dx = torch.empty_like(x)
    N_total = x.numel()
    _launch_bwd(x, dy, dx, N_total, approximate=approximate)
    return dx


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS GELU_ FORWARD")
    inp = A.contiguous()
    N_total = inp.numel()
    _launch_fwd(inp, inp, N_total)
    return A
