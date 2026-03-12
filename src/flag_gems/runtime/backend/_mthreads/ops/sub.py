import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.sub import sub as default_sub
from flag_gems.ops.sub import sub_ as default_sub_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def sub_tensor_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x - y * alpha
    tl.store(out_ptr + offsets, out, mask=mask)


# Optimized inplace kernel without alpha multiplication
@libentry()
@triton.jit
def sub_tensor_tensor_inplace_no_alpha_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x - y
    tl.store(x_ptr + offsets, out, mask=mask)


# Inplace kernel with alpha multiplication
@libentry()
@triton.jit
def sub_tensor_tensor_inplace_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x - y * alpha
    tl.store(x_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def sub_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    scalar_val = scalar * alpha
    out = x - scalar_val
    tl.store(out_ptr + offsets, out, mask=mask)


# Inplace kernel for tensor - scalar (no alpha)
@libentry()
@triton.jit
def sub_tensor_scalar_inplace_no_alpha_kernel(
    x_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x - scalar
    tl.store(x_ptr + offsets, out, mask=mask)


# Inplace kernel for tensor - scalar (with alpha)
@libentry()
@triton.jit
def sub_tensor_scalar_inplace_kernel(
    x_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    scalar_val = scalar * alpha
    out = x - scalar_val
    tl.store(x_ptr + offsets, out, mask=mask)


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.numel() == 0 or not x.is_contiguous():
        return False
    return True


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


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_MTHREADS SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if not (_use_triton_kernel(A) and _use_triton_kernel(B) and A.shape == B.shape):
            return default_sub(A, B, alpha=alpha)

        if B.device != A.device:
            B = B.to(A.device)

        out = torch.empty_like(A)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(out.device):
            sub_tensor_tensor_kernel[grid](A, B, out, alpha, n_elements)
        return out
    elif isinstance(A, torch.Tensor):
        if not _use_triton_kernel(A):
            return default_sub(A, B, alpha=alpha)

        out = torch.empty_like(A)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(out.device):
            sub_tensor_scalar_kernel[grid](A, out, B, alpha, n_elements)
        return out
    elif isinstance(B, torch.Tensor):
        return default_sub(A, B, alpha=alpha)
    else:
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_MTHREADS SUB_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if not (_use_triton_kernel(A) and _use_triton_kernel(B) and A.shape == B.shape):
            return default_sub_(A, B, alpha=alpha)

        if B.device != A.device:
            B = B.to(A.device)

        n_elements = A.numel()
        BLOCK_SIZE, num_warps = _get_inplace_config(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch_device_fn.device(A.device):
            if alpha == 1:
                sub_tensor_tensor_inplace_no_alpha_kernel[grid](
                    A, B, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
                )
            else:
                sub_tensor_tensor_inplace_kernel[grid](
                    A, B, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
                )
        return A
    elif isinstance(A, torch.Tensor):
        if not _use_triton_kernel(A):
            return default_sub_(A, B, alpha=alpha)

        n_elements = A.numel()
        BLOCK_SIZE, num_warps = _get_inplace_config(n_elements)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        with torch_device_fn.device(A.device):
            if alpha == 1:
                sub_tensor_scalar_inplace_no_alpha_kernel[grid](
                    A, B, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
                )
            else:
                sub_tensor_scalar_inplace_kernel[grid](
                    A, B, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
                )
        return A
    else:
        raise ValueError("Unreachable.")
