import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# Small matrix inverse using explicit formulas (2x2 and 3x3)
@triton.jit
def inv_2x2(A, Inv, stride_a, stride_inv):
    """Inverse of 2x2 matrix: [a b; c d] -> 1/(ad-bc) * [d -b; -c a]"""
    a = tl.load(A)
    b = tl.load(A + 1)
    c = tl.load(A + stride_a)
    d = tl.load(A + stride_a + 1)

    det = a * d - b * c
    det = 1.0 / det

    tl.store(Inv, det * d)
    tl.store(Inv + 1, det * (-b))
    tl.store(Inv + stride_inv, det * (-c))
    tl.store(Inv + stride_inv + 1, det * a)


@triton.jit
def inv_3x3(A, Inv, stride_a, stride_inv):
    """Inverse of 3x3 matrix using cofactor method"""
    # Load matrix elements
    a11 = tl.load(A)
    a12 = tl.load(A + 1)
    a13 = tl.load(A + 2)
    a21 = tl.load(A + stride_a)
    a22 = tl.load(A + stride_a + 1)
    a23 = tl.load(A + stride_a + 2)
    a31 = tl.load(A + 2 * stride_a)
    a32 = tl.load(A + 2 * stride_a + 1)
    a33 = tl.load(A + 2 * stride_a + 2)

    # Compute cofactors C[row,col] = (-1)^(row+col) * M[row,col]
    # C[0,0]: remove row 0, col 0
    c00 = a22 * a33 - a23 * a32
    # C[0,1]: remove row 0, col 1
    c01 = -(a21 * a33 - a23 * a31)
    # C[0,2]: remove row 0, col 2
    c02 = a21 * a32 - a22 * a31

    # C[1,0]: remove row 1, col 0
    c10 = -(a12 * a33 - a13 * a32)
    # C[1,1]: remove row 1, col 1
    c11 = a11 * a33 - a13 * a31
    # C[1,2]: remove row 1, col 2
    c12 = -(a11 * a32 - a12 * a31)

    # C[2,0]: remove row 2, col 0
    c20 = a12 * a23 - a13 * a22
    # C[2,1]: remove row 2, col 1
    c21 = -(a11 * a23 - a13 * a21)
    # C[2,2]: remove row 2, col 2
    c22 = a11 * a22 - a12 * a21

    # Determinant using cofactor expansion along first row
    det = a11 * c00 + a12 * c01 + a13 * c02
    det = 1.0 / det

    # Store inverse: inv = adj(A) / det = C^T / det
    # adj(A)[i,j] = C[j,i] (transpose of cofactor matrix)
    tl.store(Inv, det * c00)  # inv[0,0] = C[0,0]
    tl.store(Inv + 1, det * c10)  # inv[0,1] = C[1,0]
    tl.store(Inv + 2, det * c20)  # inv[0,2] = C[2,0]
    tl.store(Inv + stride_inv, det * c01)  # inv[1,0] = C[0,1]
    tl.store(Inv + stride_inv + 1, det * c11)  # inv[1,1] = C[1,1]
    tl.store(Inv + stride_inv + 2, det * c21)  # inv[1,2] = C[2,1]
    tl.store(Inv + 2 * stride_inv, det * c02)  # inv[2,0] = C[0,2]
    tl.store(Inv + 2 * stride_inv + 1, det * c12)  # inv[2,1] = C[1,2]
    tl.store(Inv + 2 * stride_inv + 2, det * c22)  # inv[2,2] = C[2,2]


@libentry()
@triton.jit
def linalg_inv_kernel(
    A,
    Inv,
    n,
    stride_a,
    stride_inv,
    batch_size,
):
    """
    Matrix inverse kernel.
    Uses explicit formulas for 2x2 and 3x3 matrices.
    """
    pid = tle.program_id(0)
    if pid >= batch_size:
        return

    n_i = n
    a_off = pid * stride_a
    inv_off = pid * stride_inv

    # Dispatch based on matrix size
    if n_i == 2:
        inv_2x2(A + a_off, Inv + inv_off, n, n)
    elif n_i == 3:
        inv_3x3(A + a_off, Inv + inv_off, n, n)
    else:
        # This case should not be reached for valid inputs
        pass


def linalg_inv(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a square matrix or a batch of square matrices.

    Args:
        A: Input tensor of shape (*, n, n) where * is zero or more batch dimensions

    Returns:
        Inverse tensor of shape (*, n, n)
    """
    logger.debug("GEMS linalg_inv")

    if A.numel() == 0:
        return torch.empty_like(A)

    original_shape = A.shape
    n = original_shape[-1]

    if n == 0:
        return torch.empty_like(A)

    # Only support 2x2 and 3x3 matrices for now
    if n not in (2, 3):
        raise ValueError(
            f"Matrix size {n}x{n} not supported." " Only 2x2 and 3x3 are supported."
        )

    # Handle batch dimensions
    batch_dims = original_shape[:-2]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    if batch_size == 0:
        batch_size = 1

    # For numerical stability, always use float32 for computation
    input_dtype = A.dtype
    A_compute = A.to(torch.float32)

    A_flat = A_compute.reshape(batch_size, n, n)
    inv_flat = torch.empty_like(A_flat)

    # Use Triton kernel for matrix inverse
    grid = (batch_size,)
    linalg_inv_kernel[grid](
        A_flat,
        inv_flat,
        n,
        n,
        n,
        batch_size,
    )

    # Convert back to original dtype
    inv_flat = inv_flat.to(input_dtype)

    return inv_flat.reshape(original_shape)


def linalg_inv_(A: torch.Tensor) -> torch.Tensor:
    """
    In-place version of linalg_inv
    """
    logger.debug("GEMS linalg_inv_")
    result = linalg_inv(A)
    A.copy_(result)
    return A
