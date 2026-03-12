import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.clamp import clamp_min as default_clamp_min
from flag_gems.ops.clamp import clamp_min_ as default_clamp_min_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "VEC": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "VEC": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "VEC": 1}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit
def clamp_min_tensor_kernel(
    x_ptr,
    mini_ptr,
    out_ptr,
    n_elements,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    """Kernel for clamp_min with tensor min value."""
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    mini = tl.load(mini_ptr + offsets, mask=mask)

    # Compute maximum in float32 for precision
    x_f32 = x.to(tl.float32)
    mini_f32 = mini.to(tl.float32)
    out = tl.maximum(mini_f32, x_f32).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "VEC": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128, "VEC": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 2}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512, "VEC": 4}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024, "VEC": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048, "VEC": 1}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "dtype_size"],
)
@triton.jit(do_not_specialize=["mini_val"])
def clamp_min_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    mini_val,
    dtype_size,  # used for autotune key
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    """Kernel for clamp_min with scalar min value."""
    pid = tl.program_id(0)
    BLOCK_ELEMS: tl.constexpr = BLOCK_SIZE * VEC
    offsets = (pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)).to(tl.int64)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute maximum in float32 for precision
    x_f32 = x.to(tl.float32)
    out = tl.maximum(mini_val, x_f32).to(x.dtype)

    tl.store(out_ptr + offsets, out, mask=mask)


def _use_triton_kernel_tensor(A: torch.Tensor, mini: torch.Tensor) -> bool:
    """Check if we should use optimized triton kernel for tensor min."""
    if not isinstance(A, torch.Tensor) or not isinstance(mini, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if mini.device.type != "musa" or mini.dtype not in _SUPPORTED_DTYPES:
        return False
    if not A.is_contiguous() or A.numel() == 0:
        return False
    if not mini.is_contiguous() or mini.numel() != A.numel():
        return False
    return True


def _use_triton_kernel_scalar(A: torch.Tensor, mini) -> Tuple[bool, float]:
    """Check if we should use optimized triton kernel for scalar min."""
    if not isinstance(A, torch.Tensor):
        return False, 0.0
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False, 0.0
    if not A.is_contiguous() or A.numel() == 0:
        return False, 0.0
    if isinstance(mini, torch.Tensor):
        return False, 0.0
    try:
        mini_val = float(mini)
    except Exception:
        return False, 0.0
    return True, mini_val


def _launch_clamp_min_tensor(A: torch.Tensor, mini: torch.Tensor, out: torch.Tensor):
    """Launch the tensor clamp_min kernel."""
    x_flat = A.view(-1)
    mini_flat = mini.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        clamp_min_tensor_kernel[grid](
            x_flat, mini_flat, out_flat, n_elements, dtype_size
        )
    return out


def _launch_clamp_min_scalar(A: torch.Tensor, mini_val: float, out: torch.Tensor):
    """Launch the scalar clamp_min kernel."""
    x_flat = A.view(-1)
    out_flat = out.view(-1)
    n_elements = out_flat.numel()
    dtype_size = out_flat.element_size()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"] * META["VEC"]),)
    with torch_device_fn.device(out.device):
        clamp_min_scalar_kernel[grid](
            x_flat, out_flat, n_elements, mini_val, dtype_size
        )
    return out


def clamp_min(A, mini):
    """Optimized clamp_min for mthreads backend."""
    logger.debug("GEMS_MTHREADS CLAMP_MIN")

    # Check if mini is a tensor
    if isinstance(mini, torch.Tensor):
        if _use_triton_kernel_tensor(A, mini):
            out = torch.empty_like(A)
            return _launch_clamp_min_tensor(A, mini, out)
    else:
        use_triton, mini_val = _use_triton_kernel_scalar(A, mini)
        if use_triton:
            out = torch.empty_like(A)
            return _launch_clamp_min_scalar(A, mini_val, out)

    # Fallback to default implementation
    return default_clamp_min(A, mini)


def clamp_min_(A, mini):
    """Optimized in-place clamp_min_ for mthreads backend."""
    logger.debug("GEMS_MTHREADS CLAMP_MIN_")

    # Check if mini is a tensor
    if isinstance(mini, torch.Tensor):
        if _use_triton_kernel_tensor(A, mini):
            return _launch_clamp_min_tensor(A, mini, A)
    else:
        use_triton, mini_val = _use_triton_kernel_scalar(A, mini)
        if use_triton:
            return _launch_clamp_min_scalar(A, mini_val, A)

    # Fallback to default implementation
    return default_clamp_min_(A, mini)
