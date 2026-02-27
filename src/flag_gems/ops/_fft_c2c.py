import logging
import math
from typing import List

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


@libentry()
@triton.jit
def bit_reversal_permutation_kernel(
    in_real_ptr,
    in_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    batch_stride,
    n,
    log2_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Perform bit-reversal permutation for FFT."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    batch_offset = batch_id * batch_stride

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Compute bit-reversed indices
    rev_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    temp_idx = offsets.to(tl.int32)
    for _ in range(log2_n):
        rev_idx = (rev_idx << 1) | (temp_idx & 1)
        temp_idx = temp_idx >> 1

    # Load from original positions
    in_real = tl.load(in_real_ptr + batch_offset + offsets, mask=mask, other=0.0)
    in_imag = tl.load(in_imag_ptr + batch_offset + offsets, mask=mask, other=0.0)

    # Store to bit-reversed positions
    tl.store(out_real_ptr + batch_offset + rev_idx, in_real, mask=mask)
    tl.store(out_imag_ptr + batch_offset + rev_idx, in_imag, mask=mask)


@libentry()
@triton.jit
def fft_butterfly_stage_kernel(
    real_ptr,
    imag_ptr,
    twiddle_real_ptr,
    twiddle_imag_ptr,
    batch_stride,
    n,
    half_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Single butterfly stage of FFT."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    batch_offset = batch_id * batch_stride
    step = half_size * 2

    # Each thread handles one butterfly pair
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_idx = offsets  # Index of butterfly pair within batch

    # Total number of butterfly pairs
    num_pairs = n // 2
    mask = pair_idx < num_pairs

    # Calculate which group and position within group
    group = pair_idx // half_size
    pos_in_group = pair_idx % half_size

    # Upper and lower indices
    upper_idx = group * step + pos_in_group
    lower_idx = upper_idx + half_size

    # Twiddle factor index
    tw_idx = pos_in_group * (n // step)

    # Load upper and lower values
    upper_real = tl.load(real_ptr + batch_offset + upper_idx, mask=mask, other=0.0)
    upper_imag = tl.load(imag_ptr + batch_offset + upper_idx, mask=mask, other=0.0)
    lower_real = tl.load(real_ptr + batch_offset + lower_idx, mask=mask, other=0.0)
    lower_imag = tl.load(imag_ptr + batch_offset + lower_idx, mask=mask, other=0.0)

    # Load twiddle factors
    tw_real = tl.load(twiddle_real_ptr + tw_idx, mask=mask, other=1.0)
    tw_imag = tl.load(twiddle_imag_ptr + tw_idx, mask=mask, other=0.0)

    # Complex multiplication: lower * twiddle
    prod_real = lower_real * tw_real - lower_imag * tw_imag
    prod_imag = lower_real * tw_imag + lower_imag * tw_real

    # Butterfly operation
    new_upper_real = upper_real + prod_real
    new_upper_imag = upper_imag + prod_imag
    new_lower_real = upper_real - prod_real
    new_lower_imag = upper_imag - prod_imag

    # Store results
    tl.store(real_ptr + batch_offset + upper_idx, new_upper_real, mask=mask)
    tl.store(imag_ptr + batch_offset + upper_idx, new_upper_imag, mask=mask)
    tl.store(real_ptr + batch_offset + lower_idx, new_lower_real, mask=mask)
    tl.store(imag_ptr + batch_offset + lower_idx, new_lower_imag, mask=mask)


@libentry()
@triton.jit
def fft_normalize_kernel(
    real_ptr,
    imag_ptr,
    scale,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply normalization scaling to FFT output."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    real_val = tl.load(real_ptr + offsets, mask=mask, other=0.0)
    imag_val = tl.load(imag_ptr + offsets, mask=mask, other=0.0)

    tl.store(real_ptr + offsets, real_val * scale, mask=mask)
    tl.store(imag_ptr + offsets, imag_val * scale, mask=mask)


def compute_twiddle_factors(n: int, inverse: bool, device: torch.device) -> torch.Tensor:
    """
    Compute twiddle factors for FFT.
    W_n^k = exp(-2*pi*i*k/n) for forward FFT
    W_n^k = exp(2*pi*i*k/n) for inverse FFT
    """
    sign = 1.0 if inverse else -1.0
    k = torch.arange(n // 2, device=device, dtype=torch.float32)
    angle = sign * 2.0 * math.pi * k / n
    twiddle = torch.complex(torch.cos(angle), torch.sin(angle))
    return twiddle


def fft_1d_triton(
    x: torch.Tensor,
    inverse: bool = False,
    normalization: int = 0,
) -> torch.Tensor:
    """
    1D FFT using Triton with multi-stage butterfly approach.
    Only supports power-of-2 sizes.
    """
    original_shape = x.shape
    n = x.shape[-1]

    if not is_power_of_2(n):
        raise ValueError(f"FFT size must be power of 2, got {n}")

    # Flatten batch dimensions
    batch_size = x.numel() // n
    x_flat = x.reshape(batch_size, n)

    # Separate real and imaginary parts (use float32 for computation)
    if x.dtype == torch.complex64:
        compute_dtype = torch.float32
    else:  # complex128
        compute_dtype = torch.float64

    x_real = x_flat.real.to(compute_dtype).contiguous()
    x_imag = x_flat.imag.to(compute_dtype).contiguous()

    # Allocate output buffers
    out_real = torch.empty_like(x_real)
    out_imag = torch.empty_like(x_imag)

    log2_n = int(math.log2(n))
    BLOCK_SIZE = min(1024, n)

    with torch_device_fn.device(x.device):
        # Step 1: Bit-reversal permutation
        grid_perm = (batch_size, triton.cdiv(n, BLOCK_SIZE))
        bit_reversal_permutation_kernel[grid_perm](
            x_real,
            x_imag,
            out_real,
            out_imag,
            n,  # batch_stride
            n,
            log2_n,
            BLOCK_SIZE,
        )

        # Compute twiddle factors
        twiddle = compute_twiddle_factors(n, inverse, x.device)
        twiddle_real = twiddle.real.to(compute_dtype).contiguous()
        twiddle_imag = twiddle.imag.to(compute_dtype).contiguous()

        # Step 2: Butterfly stages
        num_pairs = n // 2
        half_size = 1
        for _ in range(log2_n):
            grid_butterfly = (batch_size, triton.cdiv(num_pairs, BLOCK_SIZE))
            fft_butterfly_stage_kernel[grid_butterfly](
                out_real,
                out_imag,
                twiddle_real,
                twiddle_imag,
                n,  # batch_stride
                n,
                half_size,
                BLOCK_SIZE,
            )
            half_size *= 2

        # Step 3: Apply normalization
        # normalization=0 (backward): inverse gets 1/N, forward gets 1
        # normalization=1 (ortho): both get 1/sqrt(N)
        # normalization=2 (forward): forward gets 1/N, inverse gets 1
        if normalization == 0:  # backward normalization
            scale = 1.0 / n if inverse else 1.0
        elif normalization == 1:  # ortho normalization
            scale = 1.0 / math.sqrt(n)
        elif normalization == 2:  # forward normalization
            scale = 1.0 / n if not inverse else 1.0
        else:
            scale = 1.0

        if scale != 1.0:
            numel = batch_size * n
            grid_norm = (triton.cdiv(numel, BLOCK_SIZE),)
            fft_normalize_kernel[grid_norm](
                out_real,
                out_imag,
                scale,
                numel,
                BLOCK_SIZE,
            )

    # Combine real and imaginary parts
    output = torch.complex(out_real, out_imag)
    output = output.reshape(original_shape)

    # Convert back to original dtype if needed
    if output.dtype != x.dtype:
        output = output.to(x.dtype)

    return output


def _fft_c2c_single_dim(
    x: torch.Tensor,
    dim: int,
    normalization: int,
    forward: bool,
) -> torch.Tensor:
    """
    FFT along a single dimension.
    Uses Triton for power-of-2 sizes, torch.fft for other sizes.
    """
    n = x.shape[dim]
    inverse = not forward

    # Move FFT dimension to last
    if dim != -1 and dim != x.ndim - 1:
        x = x.movedim(dim, -1)
        moved = True
    else:
        moved = False

    # Use Triton implementation for power-of-2 sizes
    if is_power_of_2(n) and n >= 2:
        result = fft_1d_triton(x, inverse=inverse, normalization=normalization)
    else:
        # Fall back to torch.fft for non-power-of-2 sizes
        # This avoids recursion since torch.fft.fft/ifft use a different code path
        norm_map = {0: None, 1: "ortho", 2: "forward"}
        norm_str = norm_map.get(normalization, None)
        if forward:
            result = torch.fft.fft(x, dim=-1, norm=norm_str)
        else:
            result = torch.fft.ifft(x, dim=-1, norm=norm_str)

    # Move dimension back
    if moved:
        result = result.movedim(-1, dim)

    return result


def _fft_c2c(
    self: torch.Tensor,
    dim: List[int],
    normalization: int,
    forward: bool,
) -> torch.Tensor:
    """
    Complex-to-complex FFT.

    Args:
        self: Input complex tensor
        dim: Dimensions along which to compute FFT
        normalization: 0 = no normalization, 1 = ortho, 2 = forward
        forward: True for forward FFT, False for inverse FFT

    Returns:
        Complex tensor with FFT applied
    """
    logger.debug("GEMS _FFT_C2C")

    # Ensure input is complex
    if not self.is_complex():
        self = torch.complex(self, torch.zeros_like(self))

    # Make contiguous
    self = self.contiguous()

    # Normalize dimensions
    ndim = self.ndim
    dim = [(d % ndim) for d in dim]

    # Apply FFT along each dimension
    result = self
    for d in dim:
        result = _fft_c2c_single_dim(result, d, normalization, forward)

    return result
