import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def true_divide_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x / y
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_divide_scalar_kernel(
    x_ptr,
    y_scalar,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = x / y_scalar
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_divide_scalar_tensor_kernel(
    x_scalar,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute: x_scalar / y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask)
    result = x_scalar / y
    tl.store(output_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_divide__kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place true divide: x = x / y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x / y
    tl.store(x_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def true_divide__scalar_kernel(
    x_ptr,
    y_scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place true divide with scalar: x = x / y_scalar"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    result = x / y_scalar
    tl.store(x_ptr + offsets, result, mask=mask)


def true_divide(A, B):
    """True divide operator: output = A / B

    Supports:
    - tensor / tensor
    - tensor / scalar
    - scalar / tensor
    - scalar / scalar (returns tensor)
    """
    logger.debug("GEMS_ILUVATAR TRUE_DIVIDE")

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

        # Handle 0-dimensional tensor (scalar tensor) - use Python computation
        if A.dim() == 0:
            result = A.item() / B.item()
            return torch.tensor(result, dtype=torch.result_type(A, B), device=A.device)

        output = torch.empty_like(A)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        true_divide_kernel[grid](A, B, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

    # A is tensor, B is scalar
    elif isinstance(A, torch.Tensor):
        A = A.contiguous()
        n_elements = A.numel()

        if n_elements == 0:
            return torch.empty_like(A)

        # Handle 0-dimensional tensor
        if A.dim() == 0:
            result = A.item() / float(B)
            result_type = torch.result_type(A, torch.tensor(B))
            return torch.tensor(result, dtype=result_type, device=A.device)

        output = torch.empty_like(A)
        y_scalar = float(B)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        true_divide_scalar_kernel[grid](
            A, y_scalar, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
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
            result = float(A) / B.item()
            result_type = torch.result_type(torch.tensor(A), B)
            return torch.tensor(result, dtype=result_type, device=B.device)

        output = torch.empty_like(B)
        x_scalar = float(A)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        true_divide_scalar_tensor_kernel[grid](
            x_scalar, B, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output

    # Both scalars
    else:
        return torch.tensor(A / B)


def true_divide_out(A, B, out):
    """True divide operator with output: out = A / B"""
    logger.debug("GEMS_ILUVATAR TRUE_DIVIDE OUT")

    result = true_divide(A, B)
    if out is not None:
        out.copy_(result)
        return out
    return result


def true_divide_(A, B):
    """In-place true divide operator: A = A / B"""
    logger.debug("GEMS_ILUVATAR TRUE_DIVIDE_")

    n_elements = A.numel()

    # Empty tensor protection
    if n_elements == 0:
        return A

    # Handle 0-dimensional tensor
    if A.dim() == 0:
        if isinstance(B, torch.Tensor):
            result = A.item() / B.item()
        else:
            result = A.item() / float(B)
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
        true_divide__kernel[grid](A, B, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    else:
        y_scalar = float(B)
        A = A.contiguous()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        true_divide__scalar_kernel[grid](A, y_scalar, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return A
