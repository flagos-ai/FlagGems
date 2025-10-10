import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')


@libentry()
@triton.jit(do_not_specialize=["value_scalar"])
def fill_scalar_kernel(
    out_ptr,
    N,
    value_scalar,
    BLOCK_SIZE: tl.constexpr,
    SUBBLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_offset = pid * BLOCK_SIZE
    cols = tl.arange(0, SUBBLOCK_SIZE)
    num_loop = triton.cdiv(BLOCK_SIZE, SUBBLOCK_SIZE)
    for iloop in tl.range(num_loop):
        offset = pid_offset + iloop * SUBBLOCK_SIZE + cols
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


@libentry()
@triton.jit
def fill_tensor_kernel(
    out_ptr,
    N,
    value_ptr,
    BLOCK_SIZE: tl.constexpr,
    SUBBLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_offset = pid * BLOCK_SIZE
    cols = tl.arange(0, SUBBLOCK_SIZE)
    num_loop = triton.cdiv(BLOCK_SIZE, SUBBLOCK_SIZE)
    for iloop in tl.range(num_loop):
        offset = pid_offset + iloop * SUBBLOCK_SIZE + cols
        value_scalar = tl.load(value_ptr)  # load the value from the tensor.
        tl.store(out_ptr + offset, value_scalar, mask=offset < N)


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    N = out.numel()
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(input.device):
        fill_tensor_kernel[grid,](out, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out


def fill_scalar(input, value):
    logger.debug("GEMS FILL")
    out = torch.empty_like(input)
    N = out.numel()
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(input.device):
        fill_scalar_kernel[grid,](out, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    N = self.numel()
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(self.device):
        fill_tensor_kernel[grid,](self, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return self


def fill_scalar_(self, value):
    logger.debug("GEMS FILL_SCALAR_")
    N = self.numel()
    # FIXME: 910B3&910B4 have 40 AIV cores while 910B1 has 50, 910B2 has 48.
    grid = min(40, N)
    BLOCK_SIZE = (N + grid - 1) // grid
    SUBBLOCK_SIZE = min(8192, BLOCK_SIZE)

    with torch_device_fn.device(self.device):
        fill_scalar_kernel[grid,](self, N, value, BLOCK_SIZE, SUBBLOCK_SIZE)
    return self
