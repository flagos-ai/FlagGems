import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Dispatch keyset for fallback to native implementation
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@libentry()
@triton.jit
def irfft_small_kernel(
    output_ptr,
    input_real_ptr,
    input_imag_ptr,
    n_out,
    n_in,
    batch_stride_out,
    batch_stride_in,
    norm_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Small IRFFT kernel for when output size fits in a single block.
    Computes the inverse real FFT using direct DFT formula.

    For IRFFT, the input is the half-Hermitian spectrum (n_in = n_out // 2 + 1).
    We reconstruct the full spectrum using Hermitian symmetry and compute IDFT.
    """
    pid = tl.program_id(0)  # batch index

    # Output indices
    out_offsets = tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < n_out

    # Initialize output accumulator
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Compute IDFT: x[n] = sum_{k=0}^{N-1} X[k] * exp(2*pi*i*k*n/N)
    # For IRFFT with Hermitian input:
    # - X[0] is real (DC component)
    # - X[k] for k=1..n_in-1 contributes X[k] + conj(X[N-k])
    # - If N is even, X[N/2] is real (Nyquist component)

    two_pi = 2.0 * 3.141592653589793
    n_out_f = n_out.to(tl.float32)

    # Process each frequency bin
    for k in range(0, n_in):
        k_f = k.to(tl.float32)

        # Load input (real and imaginary parts)
        in_idx = pid * batch_stride_in + k
        x_real = tl.load(input_real_ptr + in_idx).to(tl.float32)
        x_imag = tl.load(input_imag_ptr + in_idx).to(tl.float32)

        # Compute phase for each output sample
        # angle = 2*pi*k*n/N
        phase = two_pi * k_f * out_offsets.to(tl.float32) / n_out_f
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        # Contribution from X[k]: X[k] * exp(i*phase) = (x_real + i*x_imag) * (cos + i*sin)
        # Real part = x_real*cos - x_imag*sin
        contrib = x_real * cos_phase - x_imag * sin_phase

        # For k > 0 and k < n_out - k (i.e., not DC or Nyquist for even n_out),
        # add contribution from conjugate symmetric term X[N-k] = conj(X[k])
        # X[N-k] * exp(i*2*pi*(N-k)*n/N) = conj(X[k]) * exp(-i*2*pi*k*n/N)
        # = (x_real - i*x_imag) * (cos - i*sin)
        # Real part = x_real*cos - x_imag*sin (same as above!)
        # Wait, that's not right. Let me recalculate.
        #
        # Actually for IRFFT:
        # X[N-k] = conj(X[k]) due to Hermitian symmetry
        # The contribution from k and N-k combined:
        # X[k]*exp(i*2*pi*k*n/N) + X[N-k]*exp(i*2*pi*(N-k)*n/N)
        # = X[k]*exp(i*2*pi*k*n/N) + conj(X[k])*exp(-i*2*pi*k*n/N)
        # = 2*Re(X[k]*exp(i*2*pi*k*n/N))
        # = 2*(x_real*cos - x_imag*sin)

        # So for k=0: just add once (DC)
        # For k=n_out/2 if n_out is even: just add once (Nyquist)
        # For other k: multiply by 2

        is_dc = k == 0
        is_nyquist = (k == n_in - 1) and (n_out % 2 == 0)

        if is_dc:
            result = result + contrib
        elif is_nyquist:
            result = result + contrib
        else:
            result = result + 2.0 * contrib

    # Apply normalization
    result = result * norm_factor

    # Store output
    out_idx = pid * batch_stride_out + out_offsets
    tl.store(output_ptr + out_idx, result, mask=out_mask)


@libentry()
@triton.jit
def irfft_general_kernel(
    output_ptr,
    input_real_ptr,
    input_imag_ptr,
    n_out,
    n_in,
    batch_stride_out,
    batch_stride_in,
    norm_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    General IRFFT kernel that processes output in tiles.
    """
    pid_batch = tl.program_id(0)
    pid_tile = tl.program_id(1)

    # Output indices for this tile
    tile_start = pid_tile * BLOCK_SIZE
    out_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < n_out

    # Initialize output accumulator
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    two_pi = 2.0 * 3.141592653589793
    n_out_f = n_out.to(tl.float32)

    # Process each frequency bin
    for k in range(0, n_in):
        k_f = k.to(tl.float32)

        # Load input
        in_idx = pid_batch * batch_stride_in + k
        x_real = tl.load(input_real_ptr + in_idx).to(tl.float32)
        x_imag = tl.load(input_imag_ptr + in_idx).to(tl.float32)

        # Compute phase
        phase = two_pi * k_f * out_offsets.to(tl.float32) / n_out_f
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        # Contribution
        contrib = x_real * cos_phase - x_imag * sin_phase

        is_dc = k == 0
        is_nyquist = (k == n_in - 1) and (n_out % 2 == 0)

        if is_dc:
            result = result + contrib
        elif is_nyquist:
            result = result + contrib
        else:
            result = result + 2.0 * contrib

    # Apply normalization
    result = result * norm_factor

    # Store output
    out_idx = pid_batch * batch_stride_out + out_offsets
    tl.store(output_ptr + out_idx, result, mask=out_mask)


def _get_norm_factor(n: int, norm: Optional[str]) -> float:
    """Get the normalization factor for IRFFT."""
    if norm is None or norm == "backward":
        return 1.0 / n
    elif norm == "forward":
        return 1.0
    elif norm == "ortho":
        return 1.0 / math.sqrt(n)
    else:
        raise ValueError(f"Invalid norm mode: {norm}")


def fft_irfft(
    input: torch.Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the inverse of torch.fft.rfft.

    Args:
        input: Complex input tensor representing a half-Hermitian signal
        n: Output signal length. Defaults to 2*(input.size(dim) - 1)
        dim: Dimension along which to compute the IRFFT
        norm: Normalization mode ("backward", "forward", "ortho")

    Returns:
        Real tensor with the inverse FFT result
    """
    logger.debug("GEMS FFT_IRFFT")

    # Handle dimension
    dim = dim if dim >= 0 else input.ndim + dim

    # Input size along the transform dimension
    input_size = input.size(dim)

    # Determine output size
    if n is None:
        n_out = 2 * (input_size - 1)
    else:
        n_out = n

    # For small sizes only, use Triton implementation
    # For larger sizes, use PyTorch's cuFFT which is both faster and more accurate
    # The direct DFT approach in Triton has O(N^2) complexity and accumulates
    # numerical errors for larger sizes
    use_triton = (
        input.is_contiguous() and
        dim == input.ndim - 1 and
        n_out <= 256 and  # Only use Triton for small sizes to maintain precision
        input_size == n_out // 2 + 1
    )

    if not use_triton:
        # Fall back to PyTorch's native implementation using redispatch
        return torch.ops.aten.fft_irfft.default.redispatch(
            _FALLBACK_KEYSET, input, n_out, dim, norm
        )

    # Get normalization factor
    norm_factor = _get_norm_factor(n_out, norm)

    # Prepare input - extract real and imaginary parts
    input_real = input.real.contiguous()
    input_imag = input.imag.contiguous()

    # Determine output dtype based on input
    if input.dtype == torch.complex64:
        out_dtype = torch.float32
    elif input.dtype == torch.complex128:
        out_dtype = torch.float64
    else:
        out_dtype = torch.float32

    # Calculate batch size (product of all dimensions except the transform dimension)
    batch_shape = list(input.shape)
    batch_shape.pop(dim)
    batch_size = 1
    for s in batch_shape:
        batch_size *= s

    # Reshape for batch processing
    # Move transform dimension to last
    if dim != input.ndim - 1:
        input_real = input_real.movedim(dim, -1)
        input_imag = input_imag.movedim(dim, -1)

    # Flatten batch dimensions
    input_real_flat = input_real.reshape(-1, input_size)
    input_imag_flat = input_imag.reshape(-1, input_size)

    # Allocate output
    output_flat = torch.empty((batch_size, n_out), dtype=out_dtype, device=input.device)

    # Compute strides
    batch_stride_in = input_size
    batch_stride_out = n_out

    # Choose block size
    BLOCK_SIZE = triton.next_power_of_2(n_out)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048

    # Launch kernel
    if n_out <= BLOCK_SIZE:
        # Single tile per batch
        grid = (batch_size,)
        with torch_device_fn.device(input.device):
            irfft_small_kernel[grid](
                output_flat,
                input_real_flat,
                input_imag_flat,
                n_out,
                input_size,
                batch_stride_out,
                batch_stride_in,
                norm_factor,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    else:
        # Multiple tiles per batch
        num_tiles = triton.cdiv(n_out, BLOCK_SIZE)
        grid = (batch_size, num_tiles)
        with torch_device_fn.device(input.device):
            irfft_general_kernel[grid](
                output_flat,
                input_real_flat,
                input_imag_flat,
                n_out,
                input_size,
                batch_stride_out,
                batch_stride_in,
                norm_factor,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    # Reshape output back to original batch shape
    output_shape = list(input.shape)
    output_shape[dim] = n_out

    # Unflatten and move dimension back
    output = output_flat.reshape(batch_shape + [n_out])
    if dim != input.ndim - 1:
        output = output.movedim(-1, dim)

    return output.reshape(output_shape)
