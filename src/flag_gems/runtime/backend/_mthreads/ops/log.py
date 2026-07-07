import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.log import log as default_log  # fallback
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_lang_helper import tl_extra_shim

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
    key=["n_elements", "dtype_size"],
)
@triton.jit
def log_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,
    BLOCK_SIZE: tl.constexpr,
    USE_APPROX: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    if USE_APPROX:
        pos_mask = x_fp32 > 0
        zero_mask = x_fp32 == 0
        ix = x_fp32.to(tl.int32, bitcast=True)
        exp = (ix >> 23) & 0xFF
        mant = (ix & 0x7FFFFF) | 0x3F800000
        m = mant.to(tl.float32, bitcast=True)
        k = exp.to(tl.int32) - 127
        t = (m - 1.0) / (m + 1.0)
        t2 = t * t
        # t4 = t2 * t2
        # t6 = t4 * t2
        log_m = 2.0 * (t + t2 * t * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0))))
        log_val = log_m + k.to(tl.float32) * 0.6931471805599453
        nan_or_inf = tl.where(zero_mask, -float("inf"), float("nan"))
        y = tl.where(pos_mask, log_val, nan_or_inf)
    else:
        y = tl_extra_shim.log(x_fp32)
    tl.store(out_ptr + offsets, y, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    return True, x.element_size()


def _launch_log(x: torch.Tensor, out: torch.Tensor, dtype_size: int):
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        log_kernel[grid](x, out, n_elements, dtype_size, USE_APPROX=dtype_size == 2)
    return out


def log(x):
    logger.debug("GEMS_MTHREADS LOG")
    use_triton, dtype_size = _use_triton_kernel(x)
    if not use_triton:
        return default_log(x)

    out = torch.empty_like(x)
    return _launch_log(x, out, dtype_size)
