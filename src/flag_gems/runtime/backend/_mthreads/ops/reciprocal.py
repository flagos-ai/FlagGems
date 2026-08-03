import logging
from typing import Tuple

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
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def reciprocal_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CLAMP_INF: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    y = 1.0 / x_fp32
    if CLAMP_INF:
        # The fp32 -> fp16/bf16 store saturates to the dtype max instead of
        # producing inf on overflow, which diverges from PyTorch's IEEE
        # rounding (and the CPU reference). Force a proper inf here.
        y = tl.where(y > DTYPE_MAX, float("inf"), y)
        y = tl.where(y < -DTYPE_MAX, float("-inf"), y)
    tl.store(out_ptr + offsets, y, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, bool, float]:
    if not isinstance(x, torch.Tensor):
        return False, False, 0.0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, False, 0.0
    if x.numel() == 0 or not x.is_contiguous():
        return False, False, 0.0
    # Only fp16/bf16 reciprocals can overflow the destination dtype; clamp them.
    clamp_inf = x.element_size() == 2
    dtype_max = torch.finfo(x.dtype).max if clamp_inf else 0.0
    return True, clamp_inf, dtype_max


def _launch_reciprocal(
    x: torch.Tensor, out: torch.Tensor, clamp_inf: bool, dtype_max: float
):
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        reciprocal_kernel[grid](
            x, out, n_elements, CLAMP_INF=clamp_inf, DTYPE_MAX=dtype_max
        )
    return out


def reciprocal(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL")
    use_triton, clamp_inf, dtype_max = _use_triton_kernel(A)
    if not use_triton:
        return default_reciprocal(A)

    out = torch.empty_like(A)
    return _launch_reciprocal(A, out, clamp_inf, dtype_max)


def reciprocal_(A):
    logger.debug("GEMS_MTHREADS RECIPROCAL_")
    use_triton, clamp_inf, dtype_max = _use_triton_kernel(A)
    if not use_triton:
        return default_reciprocal_(A)

    return _launch_reciprocal(A, A, clamp_inf, dtype_max)
