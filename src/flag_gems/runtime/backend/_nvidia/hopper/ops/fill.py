import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def fill_scalar_kernel(
    out_ptr,
    value_scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load a dummy value to infer the dtype of out_ptr
    dummy = tl.load(out_ptr + offsets, mask=mask, other=0)
    fill_val = tl.full([BLOCK_SIZE], value_scalar, dtype=dummy.dtype)
    tl.store(out_ptr + offsets, fill_val, mask=mask)


@triton.jit
def fill_tensor_kernel(
    out_ptr,
    value_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(value_ptr)
    tl.store(out_ptr + offsets, val, mask=mask)


def fill_scalar(input, value):
    logger.debug("GEMS_HOPPER FILL_SCALAR")
    out = torch.empty_like(input)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_HOPPER FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_HOPPER FILL_TENSOR")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_HOPPER FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    n_elements = out.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid](out, value, n_elements, BLOCK_SIZE=1024)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS FILL_TENSOR_ HOPPER")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    n_elements = self.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        fill_tensor_kernel[grid](self, value, n_elements, BLOCK_SIZE=1024)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_ HOPPER")
    n_elements = self.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid](self, value, n_elements, BLOCK_SIZE=1024)
    return self
