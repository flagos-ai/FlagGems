import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


@triton.jit
def lt_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 16,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    y_vals = tl.load(y_ptr + offsets, mask=mask)

    out = x_vals < y_vals
    tl.store(out_ptr + offsets, out, mask=mask)


def lt_block(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on same device and dtype
    assert x.device == y.device, "Tensors must be on the same device"
    # device = x.device
    dtype = torch.float32

    # Broadcast tensors to the same shape
    x_b, y_b = torch.broadcast_tensors(x.to(dtype), y.to(dtype))
    out = torch.empty_like(x_b, dtype=torch.bool)

    # Flatten tensors for Triton kernel
    x_b_flat = x_b.contiguous().view(-1)
    y_b_flat = y_b.contiguous().view(-1)
    out_flat = out.view(-1)

    n_elements = out_flat.numel()

    # Launch Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    lt_kernel[grid](x_b_flat, y_b_flat, out_flat, n_elements, BLOCK_SIZE=16)

    return out


def lt(A, B):
    logging.debug("GEMS LT")
    return lt_block(A, B)
    # return lt_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logging.debug("GEMS LT SCALAR")
    return lt_func_scalar(A, B)
