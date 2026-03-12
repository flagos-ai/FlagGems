import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.sqrt import sqrt as default_sqrt
from flag_gems.ops.sqrt import sqrt_ as default_sqrt_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def sqrt_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # tl.sqrt only supports fp32/fp64, convert to fp32 first
    x_fp32 = x.to(tl.float32)
    out_fp32 = tl.sqrt(x_fp32)
    out = out_fp32.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)


# Inplace kernel without autotune to avoid data corruption during tuning
@libentry()
@triton.jit
def sqrt_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    out_fp32 = tl.sqrt(x_fp32)
    out = out_fp32.to(x.dtype)
    tl.store(x_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    return True, x.element_size()


def _launch_sqrt(x: torch.Tensor, out: torch.Tensor, dtype_size: int):
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        sqrt_kernel[grid](x, out, n_elements, dtype_size)
    return out


def _get_inplace_config(n_elements: int):
    """Select optimal BLOCK_SIZE and num_warps based on tensor size."""
    if n_elements <= 1024:
        return 256, 4
    elif n_elements <= 65536:
        return 512, 8
    elif n_elements <= 1048576:  # 1M
        return 1024, 8
    elif n_elements <= 16777216:  # 16M
        return 2048, 16
    else:
        return 4096, 32


def _launch_sqrt_inplace(x: torch.Tensor):
    n_elements = x.numel()
    BLOCK_SIZE, num_warps = _get_inplace_config(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(x.device):
        sqrt_inplace_kernel[grid](
            x, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
    return x


def sqrt(x):
    logger.debug("GEMS_MTHREADS SQRT")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_sqrt(x)

    out = torch.empty_like(x)
    return _launch_sqrt(x, out, dtype_size)


def sqrt_(x):
    logger.debug("GEMS_MTHREADS SQRT_")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_sqrt_(x)

    return _launch_sqrt_inplace(x)
