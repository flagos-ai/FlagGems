import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.reciprocal import reciprocal as default_reciprocal
from flag_gems.ops.reciprocal import reciprocal_ as default_reciprocal_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def reciprocal_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Reciprocal kernel: compute 1/x"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute 1/x in float32 for precision
    result = (1.0 / x.to(tl.float32)).to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


def _use_triton_kernel(A: torch.Tensor) -> bool:
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if not A.is_contiguous() or A.numel() == 0:
        return False
    return True


def _launch_reciprocal(A: torch.Tensor, out: torch.Tensor):
    x_flat = A.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    BLOCK_SIZE = 1024
    num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(out.device):
        reciprocal_kernel[grid](x_flat, out_flat, n_elements, BLOCK_SIZE, num_warps=num_warps)
    return out


def reciprocal(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL")
    if not _use_triton_kernel(A):
        return default_reciprocal(A)

    out = torch.empty_like(A)
    return _launch_reciprocal(A, out)


def reciprocal_(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL_")
    if not _use_triton_kernel(A):
        return default_reciprocal_(A)

    return _launch_reciprocal(A, A)
