import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.mul import mul as default_mul
from flag_gems.ops.mul import mul_ as default_mul_
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
def mul_tensor_tensor_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def mul_tensor_tensor_inplace_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_last")
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy="evict_last")
    out = x * y
    tl.store(x_ptr + offsets, out, mask=mask, eviction_policy="evict_first")


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
def mul_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * scalar
    tl.store(out_ptr + offsets, out, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def mul_tensor_scalar_inplace_kernel(
    x_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy="evict_last")
    out = x * scalar
    tl.store(x_ptr + offsets, out, mask=mask, eviction_policy="evict_first")


def _use_triton_kernel(x: torch.Tensor) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.numel() == 0 or not x.is_contiguous():
        return False
    return True


def mul(A, B):
    logger.debug("GEMS_MTHREADS MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if not (_use_triton_kernel(A) and _use_triton_kernel(B) and A.shape == B.shape):
            return default_mul(A, B)

        if B.device != A.device:
            B = B.to(A.device)

        out = torch.empty_like(A)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(out.device):
            mul_tensor_tensor_kernel[grid](A, B, out, n_elements)
        return out
    elif isinstance(A, torch.Tensor):
        if not _use_triton_kernel(A):
            return default_mul(A, B)

        out = torch.empty_like(A)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(out.device):
            mul_tensor_scalar_kernel[grid](A, out, B, n_elements)
        return out
    elif isinstance(B, torch.Tensor):
        if not _use_triton_kernel(B):
            return default_mul(A, B)

        out = torch.empty_like(B)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(out.device):
            mul_tensor_scalar_kernel[grid](B, out, A, n_elements)
        return out
    else:
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_MTHREADS MUL_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if not (_use_triton_kernel(A) and _use_triton_kernel(B) and A.shape == B.shape):
            return default_mul_(A, B)

        if B.device != A.device:
            B = B.to(A.device)

        n_elements = A.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            mul_tensor_tensor_inplace_kernel[grid](A, B, n_elements)
        return A
    elif isinstance(A, torch.Tensor):
        if not _use_triton_kernel(A):
            return default_mul_(A, B)

        n_elements = A.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_device_fn.device(A.device):
            mul_tensor_scalar_inplace_kernel[grid](A, B, n_elements)
        return A
    else:
        raise ValueError("Unreachable.")
