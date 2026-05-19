import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

sqrt = tl_extra_shim.sqrt


@libentry()
@triton.jit
def linalg_eig_kernel_2x2(
    A,
    eigenvalues_real,
    eigenvalues_imag,
    eigenvectors_real,
    eigenvectors_imag,
    n,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int64)
    if pid >= batch_size:
        return

    # Stride calculations
    A = A + pid * n * n
    eigenvalues_real = eigenvalues_real + pid * n
    eigenvalues_imag = eigenvalues_imag + pid * n
    eigenvectors_real = eigenvectors_real + pid * n * n
    eigenvectors_imag = eigenvectors_imag + pid * n * n

    # Load matrix elements
    a = tl.load(A).to(tl.float32)
    b = tl.load(A + 1).to(tl.float32)
    c = tl.load(A + n).to(tl.float32)
    d = tl.load(A + n + 1).to(tl.float32)

    # Compute trace and determinant
    trace = a + d
    det = a * d - b * c

    # Compute discriminant: (a+d)^2 - 4(ad-bc) = trace^2 - 4*det
    discriminant = trace * trace - 4.0 * det

    # Compute sqrt of discriminant
    sqrt_disc = sqrt(tl.abs(discriminant))

    # Eigenvalues
    lambda1_real = (trace + sqrt_disc) / 2.0
    lambda2_real = (trace - sqrt_disc) / 2.0

    # Imaginary parts
    lambda1_imag = tl.where(discriminant < 0, sqrt(-discriminant) / 2.0, 0.0)
    lambda2_imag = tl.where(discriminant < 0, -sqrt(-discriminant) / 2.0, 0.0)

    # Store eigenvalues
    tl.store(eigenvalues_real, lambda1_real)
    tl.store(eigenvalues_real + 1, lambda2_real)
    tl.store(eigenvalues_imag, lambda1_imag)
    tl.store(eigenvalues_imag + 1, lambda2_imag)

    # Compute eigenvectors
    # For eigenvalue lambda, eigenvector v satisfies (A - lambda*I)v = 0
    # Use the first row: a*v1 + b*v2 = lambda*v1, so v = [b, lambda-a] if b != 0
    # Or use second row: c*v1 + d*v2 = lambda*v2, so v = [lambda-d, c] if c != 0
    b_nonzero = b != 0.0
    c_nonzero = c != 0.0

    # Eigenvector for lambda1
    v1_1 = tl.where(b_nonzero, b, c)
    v1_2 = tl.where(b_nonzero, lambda1_real - a, lambda1_real - d)
    # Normalize
    norm1 = sqrt(v1_1 * v1_1 + v1_2 * v1_2 + 1e-8)
    v1_1 = v1_1 / norm1
    v1_2 = v1_2 / norm1

    # Eigenvector for lambda2
    v2_1 = tl.where(b_nonzero, b, c)
    v2_2 = tl.where(b_nonzero, lambda2_real - a, lambda2_real - d)
    # Normalize
    norm2 = sqrt(v2_1 * v2_1 + v2_2 * v2_2 + 1e-8)
    v2_1 = v2_1 / norm2
    v2_2 = v2_2 / norm2

    # Handle special case when b = c = 0 (diagonal matrix)
    v1_1 = tl.where(b_nonzero | c_nonzero, v1_1, 1.0)
    v1_2 = tl.where(b_nonzero | c_nonzero, v1_2, 0.0)
    v2_1 = tl.where(b_nonzero | c_nonzero, v2_1, 0.0)
    v2_2 = tl.where(b_nonzero | c_nonzero, v2_2, 1.0)

    # Store eigenvectors (complex format with real and imag parts)
    tl.store(eigenvectors_real, v1_1)
    tl.store(eigenvectors_real + 1, v1_2)
    tl.store(eigenvectors_real + n, v2_1)
    tl.store(eigenvectors_real + n + 1, v2_2)

    # Imaginary parts are zero for real matrices
    tl.store(eigenvectors_imag, 0.0)
    tl.store(eigenvectors_imag + 1, 0.0)
    tl.store(eigenvectors_imag + n, 0.0)
    tl.store(eigenvectors_imag + n + 1, 0.0)


def linalg_eig(A: torch.Tensor):
    """
    Compute the eigenvalue decomposition of a square matrix.

    Args:
        A: A square matrix of shape (..., n, n)

    Returns:
        A named tuple (eigenvalues, eigenvectors) where:
        - eigenvalues: Complex tensor of shape (..., n)
        - eigenvectors: Complex tensor of shape (..., n, n)
    """
    logger.debug("GEMS LINALG_EIG")

    # Check input dimensions
    if A.ndim < 2:
        raise ValueError(f"linalg_eig: Expected at least 2D input, got {A.ndim}D")

    # Get the matrix dimensions
    *batch_dims, n, m = A.shape
    if n != m:
        raise ValueError(f"linalg_eig: Expected square matrix, got {A.shape}")

    if n == 0:
        raise ValueError(f"linalg_eig: Matrix size must be > 0, got {n}")

    # Flatten batch dimensions for processing
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim

    # For matrices larger than 2x2, use torch.linalg.eig
    # This is because implementing eigenvalue decomposition from scratch in Triton
    # is complex and requires iterative algorithms (QR iteration, etc.)
    # Use torch.ops.aten.linalg_eig to avoid recursion through flag_gems
    if n > 2:
        # Use torch for larger matrices - this delegates to cuSOLVER
        # Use torch.ops.aten to bypass flag_gems dispatcher and avoid recursion
        A_float = A.to(torch.float32)
        result = torch.ops.aten.linalg_eig(A_float)
        return result

    # For 2x2 matrices, use Triton kernel
    # Ensure input is contiguous and float32 for computation
    A_input = A.to(torch.float32).contiguous()

    # Allocate output tensors
    eigenvalues_real = torch.empty(
        (*batch_dims, n), dtype=torch.float32, device=A.device
    )
    eigenvalues_imag = torch.empty(
        (*batch_dims, n), dtype=torch.float32, device=A.device
    )
    eigenvectors_real = torch.empty(
        (*batch_dims, n, n), dtype=torch.float32, device=A.device
    )
    eigenvectors_imag = torch.empty(
        (*batch_dims, n, n), dtype=torch.float32, device=A.device
    )

    with torch_device_fn.device(A.device):
        # Launch kernel for 2x2 matrices
        grid = lambda META: (batch_size,)
        linalg_eig_kernel_2x2[grid](
            A_input.view(-1, n, n),
            eigenvalues_real.view(-1, n),
            eigenvalues_imag.view(-1, n),
            eigenvectors_real.view(-1, n, n),
            eigenvectors_imag.view(-1, n, n),
            n,
            batch_size,
            1,  # BLOCK_SIZE - not used in current implementation
        )

    # Combine real and imaginary parts into complex tensors
    eigenvalues = torch.complex(eigenvalues_real, eigenvalues_imag)
    eigenvectors = torch.complex(eigenvectors_real, eigenvectors_imag)

    # Return as a named tuple
    return (eigenvalues, eigenvectors)
