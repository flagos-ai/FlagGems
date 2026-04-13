import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=2),
    ],
    key=["N"],
)
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
    offset = tl.arange(0, BLOCK_MID)
    mid = mid_ptr + offset
    mask = offset < M
    mid_val = tl.load(mid, mask=mask, other=0.0)
    out_val = tl.sum(mid_val)
    tl.store(out_ptr, out_val)


def dot(x, y):
    logger.debug("Triton Dot Product")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]

    if N >= 4096:
        # Allocate for worst case (smallest block size = 2048)
        max_mid_size = triton.cdiv(N, 2048)
        block_mid = triton.next_power_of_2(max_mid_size)

        grid_1 = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

        mid = torch.empty((max_mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel_1[grid_1](x, y, mid, N)
            dot_kernel_2[(1,)](mid, out, max_mid_size, block_mid)

    else:
        block_size = triton.next_power_of_2(N)

        grid = (1, 1, 1)

        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel[grid](x, y, out, N, block_size)
            out = out.to(x.dtype)

    return out
