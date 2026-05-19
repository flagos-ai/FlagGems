import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def linalg_eigvals_kernel(
    A,
    O,
    n,
    batch_size,
    stride_a,  # stride for batch dimension
    stride_a2,  # stride for matrix row (inner dimension)
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for computing eigenvalues of small matrices.

    For 2x2 matrices, we use the analytical quadratic formula.
    For larger matrices, we delegate to torch (this kernel processes only 2x2).
    """
    pid = tl.program_id(0)
    batch_idx = pid

    if batch_idx >= batch_size:
        return

    # Compute base offset for this batch
    base_offset = batch_idx * stride_a

    # Get pointer to this batch's matrix
    a_ptr = A + base_offset

    # For 2x2 matrices: eigenvalues of [[a, b], [c, d]] are:
    # lambda = ((a+d) +/- sqrt((a+d)^2 - 4(ad-bc))) / 2
    # Compute trace = a + d
    # Compute det = ad - bc

    # Load matrix elements (2x2) - row major
    a_val = tl.load(a_ptr)  # a[0,0]
    b_val = tl.load(a_ptr + 1)  # a[0,1]
    c_val = tl.load(a_ptr + stride_a2)  # a[1,0] (next row)
    d_val = tl.load(a_ptr + stride_a2 + 1)  # a[1,1]

    # Compute trace and determinant
    trace = a_val + d_val
    det = a_val * d_val - b_val * c_val

    # Compute discriminant: tr^2 - 4*det
    discriminant = trace * trace - 4.0 * det

    # Compute sqrt of absolute discriminant
    abs_disc = tl.abs(discriminant)
    sqrt_abs_disc = tl.sqrt(abs_disc)

    # Check if eigenvalues are complex (negative discriminant)
    is_complex = discriminant < 0.0

    # For real eigenvalues (discriminant >= 0):
    #   lambda1 = (trace + sqrt(disc)) / 2
    #   lambda2 = (trace - sqrt(disc)) / 2
    # For complex eigenvalues (discriminant < 0):
    #   lambda1 = trace/2 + i * sqrt(|disc|) / 2
    #   lambda2 = trace/2 - i * sqrt(|disc|) / 2
    #   Both have real part = trace/2

    # Real part is always trace/2 (for both real and complex cases)
    real_part = trace / 2.0

    # For real eigenvalues, compute sqrt(disc); for complex, use sqrt(|disc|)
    sqrt_disc = tl.where(is_complex, sqrt_abs_disc, tl.sqrt(discriminant))

    # Eigenvalue 1
    real1 = tl.where(is_complex, real_part, (trace + sqrt_disc) / 2.0)
    imag1 = tl.where(is_complex, sqrt_abs_disc / 2.0, 0.0)

    # Eigenvalue 2
    real2 = tl.where(is_complex, real_part, (trace - sqrt_disc) / 2.0)
    imag2 = tl.where(is_complex, -sqrt_abs_disc / 2.0, 0.0)

    # Store eigenvalues as complex numbers (real, imag, real, imag, ...)
    out_ptr = O + batch_idx * stride_out
    tl.store(out_ptr, real1)
    tl.store(out_ptr + 1, imag1)
    tl.store(out_ptr + 2, real2)
    tl.store(out_ptr + 3, imag2)


def linalg_eigvals(A: torch.Tensor) -> torch.Tensor:
    """Compute the eigenvalues of a square matrix.

    This function computes the eigenvalues of a square matrix.
    For a batch of matrices, computes eigenvalues for each matrix in the batch.

    Args:
        A: Tensor of shape (*, n, n) where * is zero or more batch dimensions.

    Returns:
        Complex-valued tensor of shape (*, n) containing the eigenvalues.
        The eigenvalues are not guaranteed to be in any specific order.
    """
    logger.debug("GEMS LINALG_EIGVALS")

    # Handle input shape
    original_shape = A.shape
    if original_shape[-1] != original_shape[-2]:
        raise ValueError("linalg_eigvals: input matrix must be square")

    n = original_shape[-1]

    # For 2x2 matrices, use Triton kernel
    if n == 2:
        # Flatten batch dimensions
        batch_dims = original_shape[:-2]
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Reshape to (batch_size, 2, 2)
        A_flat = A.reshape(-1, 2, 2)

        # Use input dtype for computation - for float32 use float32, for float64 use float64
        compute_dtype = A.dtype
        output = torch.empty((batch_size, 4), dtype=compute_dtype, device=A.device)

        # Calculate strides
        stride_a = A_flat.stride(0)  # stride for batch dimension
        stride_a2 = A_flat.stride(1)  # stride for row (should be 2)
        stride_out = output.stride(0)

        # Launch kernel - pass tensors, not data_ptr()
        grid = (batch_size,)
        with torch.device(A.device):
            linalg_eigvals_kernel[grid](
                A_flat,
                output,
                n,
                batch_size,
                stride_a,
                stride_a2,
                stride_out,
                BLOCK_SIZE=32,
            )

        # Reshape output to (batch_size, 2) complex
        output_complex = torch.complex(output[:, 0::2], output[:, 1::2])
        return output_complex.reshape(*batch_dims, 2)

    # For non-2x2 matrices, raise an error (only 2x2 is currently supported)
    raise ValueError(
        f"linalg_eigvals: Only 2x2 matrices are currently supported. Got {n}x{n} matrix."
    )
