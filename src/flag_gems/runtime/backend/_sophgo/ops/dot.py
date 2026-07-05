import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Keep the large-N cap from sophgo_backend_opt to avoid local-memory pressure in
# PPL AddressAssign, while using the migrated narrow reducer for the hot path.
MAX_BLOCK = 4096


def dot_block_size(N):
    if N >= 1024:
        return 1024
    if N >= 256:
        return 256
    return triton.next_power_of_2(N)


@libentry()
@triton.jit
def dot_reduce_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < N
        a = tl.load(a_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += a * b

    total = tl.sum(acc[None, :], axis=1)
    out_idx = tl.arange(0, 1)
    tl.store(out_ptr + out_idx, total, mask=out_idx < 1)


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
    acc = tl.zeros([BLOCK_MID], dtype=tl.float32)
    for off in range(0, M, BLOCK_MID):
        offsets = off + tl.arange(0, BLOCK_MID)
        mask = offsets < M
        mid_val = tl.load(mid_ptr + offsets, mask=mask, other=0.0)
        acc += mid_val
    out_val = tl.sum(acc)
    tl.store(out_ptr, out_val)


def dot(x, y):
    logger.debug("GEMS SOPHGO DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]
    original_dtype = x.dtype

    if N == 0:
        return torch.zeros((), dtype=original_dtype, device=x.device)

    if x.dtype == torch.float64:
        x = x.to(torch.float32)
        y = y.to(torch.float32)

    x = x.contiguous()
    y = y.contiguous()

    if N < 4096:
        block_size = dot_block_size(N)
        out = torch.empty((1,), dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            dot_reduce_kernel[(1, 1, 1)](
                x,
                y,
                out,
                N,
                BLOCK_SIZE=block_size,
            )
        result = out[0]
    else:
        block_size = min(triton.next_power_of_2(math.ceil(math.sqrt(N))), MAX_BLOCK)
        mid_size = triton.cdiv(N, block_size)
        block_mid = min(MAX_BLOCK, triton.next_power_of_2(mid_size))

        mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
        out = torch.empty([], dtype=torch.float32, device=x.device)

        with torch_device_fn.device(x.device):
            dot_kernel_1[(mid_size, 1, 1)](x, y, mid, N, block_size)
            dot_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
        result = out

    if result.dtype != original_dtype:
        result = result.to(original_dtype)
    return result
