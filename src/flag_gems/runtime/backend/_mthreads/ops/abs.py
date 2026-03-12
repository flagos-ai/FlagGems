import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.abs import abs as default_abs
from flag_gems.ops.abs import abs_ as default_abs_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def abs_kernel(
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
    out = tl.abs(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    return True, x.element_size()


def _launch_abs(x: torch.Tensor, out: torch.Tensor, dtype_size: int):
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        abs_kernel[grid](x, out, n_elements, dtype_size)
    return out


def abs(x):
    logger.debug("GEMS_MTHREADS ABS")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_abs(x)

    out = torch.empty_like(x)
    return _launch_abs(x, out, dtype_size)


def abs_(x):
    logger.debug("GEMS_MTHREADS ABS_")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_abs_(x)

    return _launch_abs(x, x, dtype_size)
