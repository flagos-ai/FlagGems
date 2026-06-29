import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Cap block size so the live local-memory tensors fit on TPU even for very
# large N. dot_kernel_1 holds ~3*BLOCK_SIZE*4B (x, y, x*y); dot_kernel_2 holds
# ~BLOCK_MID*4B. Without this cap, N=2^30 gives block_size=next_pow2(sqrt(N))
# = 32768 and "local mem not enough" / ppl AddressAssign failure.
MAX_BLOCK = 4096


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum = tl.sum(x * y)
    tl.store(out_ptr, sum)


@libentry()
@triton.jit
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    partial_sum = tl.sum(x * y)
    tl.store(mid_ptr + pid, partial_sum)


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    # Reduce all partial sums (mid[0..M)) into out. M (= mid_size) can be very
    # large for big N, so loop over fixed BLOCK_MID chunks instead of one giant
    # tl.arange(0, M) that would overflow local memory, then sum the accumulator.
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    for off in range(0, M, BLOCK_MID):
        offsets = off + tl.arange(0, BLOCK_MID)
        mask = offsets < M
        mid_val = tl.load(mid_ptr + offsets, mask=mask, other=0.0)
        acc += mid_val
    out_val = tl.sum(acc)
    tl.store(out_ptr, out_val)


def dot(x, y):
    logger.debug("Triton Dot Product")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]

    # Only when N is less than TRITON_MAX_TENSOR_NUMEL can it be processed with a single kernel,
    # and performance is better when N < 4096
    if N >= 4096:
        block_size = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), MAX_BLOCK)

        mid_size = triton.cdiv(N, block_size)
        block_mid = min(MAX_BLOCK, triton.next_power_of_2(mid_size))

        grid_1 = (mid_size, 1, 1)
        grid_2 = (1, 1, 1)

        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel_1[grid_1](x, y, mid, N, block_size)
            dot_kernel_2[grid_2](mid, out, mid_size, block_mid)

    else:
        block_size = triton.next_power_of_2(N)

        grid = (1, 1, 1)

        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel[grid](x, y, out, N, block_size)
            out = out.to(x.dtype)

    return out
