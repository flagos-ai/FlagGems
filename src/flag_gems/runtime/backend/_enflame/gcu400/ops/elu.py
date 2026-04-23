import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit(do_not_specialize=["N_total", "alpha", "scale", "input_scale"])
def elu_kernel(x_ptr, out_ptr, N_total, alpha, scale, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        out = tl.where(
            x > 0.0,
            scale * input_scale * x,
            scale * alpha * (tl.exp(x * input_scale) - 1.0),
        )
        tl.store(out_ptr + off, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def elu_kernel_default(x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    """Specialized kernel for alpha=1.0, scale=1.0, input_scale=1.0:
    elu(x) = x if x > 0 else exp(x) - 1"""
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


def _launch_elu(inp, out, N_total, alpha, scale, input_scale):
    is_fp32 = inp.dtype == torch.float32
    is_default = (alpha == 1.0 and scale == 1.0 and input_scale == 1.0)

    if N_total <= 1024:
        BLOCK = 1024
    elif N_total <= 32768:
        BLOCK = triton.next_power_of_2(N_total)
    elif N_total <= 1048576:
        target_block = N_total // NUM_SIPS
        BLOCK = max(1024, 1 << (target_block - 1).bit_length())
        BLOCK = min(BLOCK, 65536)
    elif N_total <= 8388608:
        BLOCK = 32768
    else:
        if is_default and not is_fp32:
            BLOCK = 131072
        else:
            BLOCK = 65536

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    with torch_device_fn.device(inp.device):
        if is_default:
            elu_kernel_default[(grid_size,)](inp, out, N_total, BLOCK=BLOCK, num_warps=2)
        else:
            elu_kernel[(grid_size,)](
                inp, out, N_total, alpha, scale, input_scale, BLOCK=BLOCK, num_warps=2
            )


def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU")
    inp = A.contiguous()
    out = torch.empty_like(inp)
    _launch_elu(inp, out, inp.numel(), alpha, scale, input_scale)
    return out


def elu_(A, alpha=1.0, scale=1.0, input_scale=1.0):
    logger.debug("GEMS ELU_")
    inp = A.contiguous()
    _launch_elu(inp, inp, inp.numel(), alpha, scale, input_scale)
    return A


@libentry()
@triton.jit(do_not_specialize=["N_total", "alpha", "scale", "input_scale"])
def elu_backward_kernel(grad_ptr, x_ptr, out_ptr, N_total, alpha, scale, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        grad = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        grad_input = tl.where(
            x > 0.0,
            grad * scale * input_scale,
            grad * (scale * alpha * tl.exp(x * input_scale) * input_scale),
        )
        tl.store(out_ptr + off, grad_input, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def elu_backward_kernel_default(grad_ptr, x_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        grad = tl.load(grad_ptr + off, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
        grad_input = tl.where(x > 0.0, grad, grad * tl.exp(x))
        tl.store(out_ptr + off, grad_input, mask=mask)


def elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result):
    logger.debug("GEMS GCU400 ELU BACKWARD")
    grad = grad_output.contiguous()
    x = self_or_result.contiguous()
    out = torch.empty_like(grad)
    N_total = grad.numel()

    is_default = (alpha == 1.0 and scale == 1.0 and input_scale == 1.0) and not is_result

    if N_total <= 1024:
        BLOCK = 1024
    elif N_total <= 32768:
        BLOCK = triton.next_power_of_2(N_total)
    elif N_total <= 1048576:
        target_block = N_total // NUM_SIPS
        BLOCK = max(1024, 1 << (target_block - 1).bit_length())
        BLOCK = min(BLOCK, 65536)
    elif N_total <= 8388608:
        BLOCK = 32768
    else:
        if is_default and grad.dtype != torch.float32:
            BLOCK = 131072
        else:
            BLOCK = 65536

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 2)

    nw = 4 if N_total >= 1048576 else 2
    with torch_device_fn.device(grad.device):
        if is_default:
            elu_backward_kernel_default[(grid_size,)](
                grad, x, out, N_total, BLOCK=BLOCK, num_warps=nw
            )
        else:
            elu_backward_kernel[(grid_size,)](
                grad, x, out, N_total, alpha, scale, input_scale,
                BLOCK=BLOCK, num_warps=nw
            )
    return out
