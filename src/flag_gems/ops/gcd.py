import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def gcd_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    # Euclid algorithm using while loop
    for _ in range(64):
        x_old = x
        y_old = y
        y_nonzero = y_old != 0
        # Avoid division by zero: use y_safe = (y_old != 0) ? y_old : 1
        y_safe = tl.where(y_nonzero, y_old, 1)
        mod_result = x_old % y_safe
        y = tl.where(y_nonzero, mod_result, y_old)
        x = tl.where(y_nonzero, y_old, x_old)

    tl.store(out_ptr + offsets, x, mask=mask)


def gcd(A, B):
    device = A.device
    out_shape = torch.broadcast_shapes(A.shape, B.shape)
    out = torch.empty(out_shape, dtype=A.dtype, device=device)

    A = A.expand(out_shape).contiguous()
    B = B.expand(out_shape).contiguous()

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(device):
        gcd_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def gcd_out(A, B, out):
    device = A.device
    target_shape = out.shape
    A = A.expand(target_shape).contiguous()
    B = B.expand(target_shape).contiguous()

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(device):
        gcd_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
