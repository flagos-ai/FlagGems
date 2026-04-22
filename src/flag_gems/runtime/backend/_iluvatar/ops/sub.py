import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sub_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x - y * alpha
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def sub_scalar_kernel(
    x_ptr,
    y_scalar,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = x - y_scalar * alpha
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def sub_scalar_tensor_kernel(
    x_scalar,
    y_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute: x_scalar - y * alpha"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)
    result = x_scalar - y * alpha
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def sub__kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place subtraction: x = x - y * alpha"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x - y * alpha
    tl.store(x_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def sub__scalar_kernel(
    x_ptr,
    y_scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place subtraction with scalar: x = x - y_scalar * alpha"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = x - y_scalar * alpha
    tl.store(x_ptr + offsets, result, mask=mask)


def sub(A, B, *, alpha=1):
    """Subtraction operator: output = A - B * alpha

    Supports:
    - tensor - tensor
    - tensor - scalar
    - scalar - tensor
    - scalar - scalar
    """
    logger.debug("GEMS_ILUVATAR SUB")

    # Handle alpha
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    alpha = float(alpha)

    # Both are tensors
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        # Handle broadcasting
        A, B = torch.broadcast_tensors(A, B)
        A = A.contiguous()
        B = B.contiguous()
        n_elements = A.numel()

        # Empty tensor protection
        if n_elements == 0:
            return torch.empty_like(A)

        # Handle 0-dimensional tensor (scalar tensor) - use elementwise loop
        if A.dim() == 0:
            # For 0-dim tensors, just use Python computation
            result = A.item() - B.item() * alpha
            return torch.tensor(result, dtype=A.dtype, device=A.device)

        output = torch.empty_like(A)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_kernel[grid](A, B, output, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

    # A is tensor, B is scalar
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        n_elements = A.numel()

        if n_elements == 0:
            return torch.empty_like(A)

        # Handle 0-dimensional tensor
        if A.dim() == 0:
            result = A.item() - float(B) * alpha
            return torch.tensor(result, dtype=A.dtype, device=A.device)

        output = torch.empty_like(A)
        y_scalar = float(B)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_scalar_kernel[grid](
            A, y_scalar, output, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output

    # B is tensor, A is scalar
    elif isinstance(B, torch.Tensor):
        B = B.contiguous()
        n_elements = B.numel()

        if n_elements == 0:
            return torch.empty_like(B)

        # Handle 0-dimensional tensor
        if B.dim() == 0:
            result = float(A) - B.item() * alpha
            return torch.tensor(result, dtype=B.dtype, device=B.device)

        output = torch.empty_like(B)
        x_scalar = float(A)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub_scalar_tensor_kernel[grid](
            x_scalar, B, output, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output

    # Both scalars
    else:
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    """In-place subtraction operator: A = A - B * alpha"""
    logger.debug("GEMS_ILUVATAR SUB_")

    # Handle alpha
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    alpha = float(alpha)

    n_elements = A.numel()

    # Empty tensor protection
    if n_elements == 0:
        return A

    # Handle 0-dimensional tensor
    if A.dim() == 0:
        if isinstance(B, torch.Tensor):
            result = A.item() - B.item() * alpha
        else:
            result = A.item() - float(B) * alpha
        A.copy_(torch.tensor(result, dtype=A.dtype, device=A.device))
        return A

    BLOCK_SIZE = 2048

    if isinstance(B, torch.Tensor):
        # Handle broadcasting
        if A.shape != B.shape:
            B = B.expand_as(A)
        B = B.contiguous()
        A = A.contiguous()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub__kernel[grid](A, B, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        y_scalar = float(B)
        A = A.contiguous()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sub__scalar_kernel[grid](A, y_scalar, alpha, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return A
