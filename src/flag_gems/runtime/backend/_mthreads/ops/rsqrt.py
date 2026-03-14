import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.rsqrt import rsqrt as default_rsqrt
from flag_gems.ops.rsqrt import rsqrt_ as default_rsqrt_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
sqrt = tl_extra_shim.sqrt


@libentry()
@triton.jit
def rsqrt_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Rsqrt kernel: compute 1/sqrt(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute 1/sqrt(x) in float32 for precision
    result = (1.0 / sqrt(x.to(tl.float32))).to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


def _use_triton_kernel(A: torch.Tensor) -> bool:
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if not A.is_contiguous() or A.numel() == 0:
        return False
    return True


def _launch_rsqrt(A: torch.Tensor, out: torch.Tensor):
    x_flat = A.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    BLOCK_SIZE = 1024
    num_warps = 8
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(out.device):
        rsqrt_kernel[grid](x_flat, out_flat, n_elements, BLOCK_SIZE, num_warps=num_warps)
    return out


def rsqrt(A):
    logger.debug("GEMS_MTHREADS RSQRT")
    if not _use_triton_kernel(A):
        return default_rsqrt(A)

    out = torch.empty_like(A)
    return _launch_rsqrt(A, out)


def rsqrt_(A):
    logger.debug("GEMS_MTHREADS RSQRT_")
    if not _use_triton_kernel(A):
        return default_rsqrt_(A)

    return _launch_rsqrt(A, A)
