import logging
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.celu import celu as default_celu
from flag_gems.ops.celu import celu_ as default_celu_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
exp = tl_extra_shim.exp


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def celu_kernel_alpha1(
    x_ptr,
    out_ptr,
    n_elements,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    x_compute = x.to(tl.float32)
    neg_mask = x_compute <= 0
    exp_val = exp(tl.where(neg_mask, x_compute, 0.0))
    neg = exp_val - 1.0
    out = tl.where(neg_mask, neg, x_compute).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit(do_not_specialize=["alpha"])
def celu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    alpha,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    x_compute = x.to(tl.float32)
    alpha_val = tl.full((1,), alpha, tl.float32)
    inv_alpha = 1.0 / alpha_val
    neg_mask = x_compute <= 0
    exp_val = exp(tl.where(neg_mask, x_compute * inv_alpha, 0.0))
    neg = alpha_val * (exp_val - 1.0)
    out = tl.where(neg_mask, neg, x_compute).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _use_triton_kernel(
    A: torch.Tensor, alpha, *, is_inplace: bool
) -> Tuple[bool, float]:
    if not isinstance(A, torch.Tensor):
        return False, 0.0
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False, 0.0
    if not A.is_contiguous() or A.numel() == 0:
        return False, 0.0
    try:
        alpha_value = (
            float(alpha) if not isinstance(alpha, torch.Tensor) else float(alpha.item())
        )
    except Exception:
        return False, 0.0
    if not math.isfinite(alpha_value):
        return False, 0.0
    return True, alpha_value


def _launch_celu(A: torch.Tensor, out: torch.Tensor, alpha_value: float):
    x_flat = A.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        if alpha_value == 1.0:
            celu_kernel_alpha1[grid](x_flat, out_flat, n_elements, dtype_size)
        else:
            celu_kernel[grid](x_flat, out_flat, n_elements, alpha_value, dtype_size)
    return out


def celu(A, alpha=1.0):
    logger.debug("GEMS_MTHREADS CELU")
    use_triton, alpha_value = _use_triton_kernel(A, alpha, is_inplace=False)
    if not use_triton:
        return default_celu(A, alpha=alpha)

    out = torch.empty_like(A)
    return _launch_celu(A, out, alpha_value)


def celu_(A, alpha=1.0):
    logger.debug("GEMS_MTHREADS CELU_")
    use_triton, alpha_value = _use_triton_kernel(A, alpha, is_inplace=True)
    if not use_triton:
        return default_celu_(A, alpha=alpha)

    return _launch_celu(A, A, alpha_value)
