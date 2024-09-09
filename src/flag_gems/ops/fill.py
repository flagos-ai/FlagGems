import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


@libentry()
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    value_scalar = tl.load(value_ptr)  # load the value from the tensor.
    tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)

    with torch.cuda.device(input.device):
        fill_tensor_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out


def fill_scalar(input, value):
    logging.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    BLOCK_SIZE = 512
    grid = triton.cdiv(N, BLOCK_SIZE)

    with torch.cuda.device(input.device):
        fill_scalar_kernel[grid,](out, N, value, BLOCK_SIZE)
    return out
