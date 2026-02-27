import logging
import math
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _real_to_complex_kernel(
    input_ptr,
    output_real_ptr,
    output_imag_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert real tensor to complex by setting imaginary part to zero."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load real values
    real_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Store to output (real part = input, imag part = 0)
    tl.store(output_real_ptr + offsets, real_vals, mask=mask)
    tl.store(output_imag_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=real_vals.dtype), mask=mask)


@libentry()
@triton.jit
def _extract_onesided_kernel(
    input_real_ptr,
    input_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    batch_size,
    input_last_dim,
    output_last_dim,
    input_batch_stride,
    output_batch_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Extract onesided FFT result (first N//2 + 1 elements of last dimension)."""
    pid = tl.program_id(0)
    batch_idx = pid // tl.cdiv(output_last_dim, BLOCK_SIZE)
    block_idx = pid % tl.cdiv(output_last_dim, BLOCK_SIZE)

    if batch_idx >= batch_size:
        return

    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_last_dim

    # Calculate input and output positions
    input_offset = batch_idx * input_batch_stride + offsets
    output_offset = batch_idx * output_batch_stride + offsets

    # Load from input
    real_vals = tl.load(input_real_ptr + input_offset, mask=mask, other=0.0)
    imag_vals = tl.load(input_imag_ptr + input_offset, mask=mask, other=0.0)

    # Store to output
    tl.store(output_real_ptr + output_offset, real_vals, mask=mask)
    tl.store(output_imag_ptr + output_offset, imag_vals, mask=mask)


def _fft_r2c(input_tensor: torch.Tensor, dim: List[int], normalization: int, onesided: bool) -> torch.Tensor:
    """
    Real-to-complex FFT implementation.

    This is the main entry point that matches torch._fft_r2c signature.

    Args:
        input_tensor: Real input tensor
        dim: List of dimensions along which to compute FFT
        normalization: Normalization mode (0=none, 1=ortho/sqrt(n), 2=forward/1/n)
        onesided: If True, return only positive frequencies

    Returns:
        Complex tensor with FFT result
    """
    logger.debug("GEMS _FFT_R2C")

    # Input validation
    assert input_tensor.is_floating_point(), "Input must be a real floating-point tensor"
    assert len(dim) > 0, "At least one dimension must be specified"

    # Normalize dimensions
    ndim = input_tensor.ndim
    dims = [(d % ndim) for d in dim]

    # Determine output dtype based on input dtype
    # float16 -> complex32 (computed in complex64, then converted)
    # float32 -> complex64
    # float64 -> complex128
    # bfloat16 -> Not supported by cuFFT, compute in complex64
    original_dtype = input_tensor.dtype
    if original_dtype == torch.float64:
        compute_dtype = torch.float64
        complex_dtype = torch.complex128
        output_complex_dtype = torch.complex128
    elif original_dtype == torch.float16:
        compute_dtype = torch.float32
        complex_dtype = torch.complex64
        output_complex_dtype = torch.complex32
    elif original_dtype == torch.bfloat16:
        compute_dtype = torch.float32
        complex_dtype = torch.complex64
        output_complex_dtype = torch.complex64  # bfloat16 -> complex64
    else:
        compute_dtype = torch.float32
        complex_dtype = torch.complex64
        output_complex_dtype = torch.complex64

    device = input_tensor.device

    with torch_device_fn.device(device):
        # Convert real input to complex (zero imaginary part)
        # This is necessary because _fft_c2c requires complex input
        input_float = input_tensor.to(compute_dtype)
        input_complex = torch.complex(
            input_float,
            torch.zeros_like(input_float)
        )

        # Use _fft_c2c for the actual FFT computation
        # _fft_c2c signature: (Tensor self, int[] dim, int normalization, bool forward) -> Tensor
        # forward=True for FFT, forward=False for IFFT
        result = torch.ops.aten._fft_c2c.default(input_complex, dims, normalization, True)

        # For onesided, extract only positive frequencies from the last dimension
        if onesided:
            # The last dimension in dims determines onesided output
            last_dim = dims[-1]
            n = input_tensor.shape[last_dim]
            onesided_n = n // 2 + 1

            # Create slice to extract onesided result
            slices = [slice(None)] * result.ndim
            slices[last_dim] = slice(0, onesided_n)
            result = result[tuple(slices)].contiguous()

        # Convert to output dtype if needed
        if result.dtype != output_complex_dtype:
            # Use view_as_real/view_as_complex to avoid FlagGems' to_copy
            # which doesn't support complex dtypes
            real_part = torch.view_as_real(result)
            if output_complex_dtype == torch.complex32:
                real_part = real_part.to(torch.float16)
            elif output_complex_dtype == torch.complex64:
                real_part = real_part.to(torch.float32)
            elif output_complex_dtype == torch.complex128:
                real_part = real_part.to(torch.float64)
            result = torch.view_as_complex(real_part.contiguous())

    return result
