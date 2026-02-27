import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}),
    ],
    key=["N", "K"],
)
@triton.jit
def _fft_c2r_kernel(
    input_real_ptr,
    input_imag_ptr,
    output_ptr,
    M,  # batch size (product of all dims except the transform dim)
    N,  # output size (last_dim_size)
    K,  # input complex size (N // 2 + 1)
    normalization: tl.constexpr,  # 0: none, 1: by_root_n, 2: by_n
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute irfft (complex-to-real FFT) using DFT matrix multiplication.

    For each output element x[n], compute:
    x[n] = X[0].real
         + 2 * sum_{k=1}^{K-2} [X[k].real * cos(2*pi*k*n/N) - X[k].imag * sin(2*pi*k*n/N)]
         + X[K-1].real * cos(pi*n)  (only if N is even)

    Then apply normalization factor based on normalization mode.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    # Precompute 2*pi/N
    two_pi_over_N = 2.0 * 3.141592653589793 / N

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # k = 0 term: X[0].real (no phase factor)
    x0_real_offset = m_offsets * K
    x0_real = tl.load(input_real_ptr + x0_real_offset, mask=m_mask, other=0.0)
    acc += x0_real[:, None]

    # k = 1 to K-2 terms: contribute twice due to conjugate symmetry
    for k in range(1, K - 1 if K > 1 else 1):
        # Load X[k].real and X[k].imag
        xk_offset = m_offsets * K + k
        xk_real = tl.load(input_real_ptr + xk_offset, mask=m_mask, other=0.0)
        xk_imag = tl.load(input_imag_ptr + xk_offset, mask=m_mask, other=0.0)

        # Compute phase: 2*pi*k*n/N
        phase = k * two_pi_over_N * n_offsets.to(tl.float32)
        cos_phase = tl.cos(phase)
        sin_phase = tl.sin(phase)

        # Contribution: 2 * [X[k].real * cos - X[k].imag * sin]
        contrib = 2.0 * (xk_real[:, None] * cos_phase[None, :] - xk_imag[:, None] * sin_phase[None, :])
        acc += contrib

    # k = K-1 term (Nyquist frequency, only if N is even)
    # If N is even, K = N//2 + 1, and X[K-1] corresponds to Nyquist
    # cos(pi*n) = (-1)^n
    if K > 1:
        xk_offset = m_offsets * K + (K - 1)
        xk_real = tl.load(input_real_ptr + xk_offset, mask=m_mask, other=0.0)

        # For Nyquist frequency: phase = pi * n, cos(pi*n) = (-1)^n
        # If N is even, include this term; if N is odd, K-1 is not exactly at Nyquist
        # and should be handled like regular terms
        is_even = (N % 2) == 0
        if is_even:
            # cos(pi * n) = (-1)^n
            n_mod2 = n_offsets % 2
            cos_nyquist = tl.where(n_mod2 == 0, 1.0, -1.0)
            acc += xk_real[:, None] * cos_nyquist[None, :]
        else:
            # N is odd, treat K-1 like a regular term with factor 2
            xk_imag = tl.load(input_imag_ptr + xk_offset, mask=m_mask, other=0.0)
            k_last = K - 1
            phase = k_last * two_pi_over_N * n_offsets.to(tl.float32)
            cos_phase = tl.cos(phase)
            sin_phase = tl.sin(phase)
            contrib = 2.0 * (xk_real[:, None] * cos_phase[None, :] - xk_imag[:, None] * sin_phase[None, :])
            acc += contrib

    # Apply normalization
    if normalization == 2:  # by_n (backward norm in torch.fft)
        acc = acc / N
    elif normalization == 1:  # by_root_n (ortho norm)
        acc = acc / tl.sqrt(N.to(tl.float32))
    # normalization == 0: no normalization (forward norm - multiply by nothing extra)

    # Store result
    output_offset = m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(output_ptr + output_offset, acc, mask=mask)


def _fft_c2r(input, dim, normalization, last_dim_size):
    """
    Compute the inverse real FFT (complex-to-real).

    Args:
        input: Complex input tensor from rfft
        dim: List of dimensions to transform (only single dimension supported)
        normalization: Normalization mode (0: none, 1: ortho, 2: backward)
        last_dim_size: Size of the output along the transform dimension

    Returns:
        Real tensor with the inverse FFT result
    """
    logger.debug("GEMS _FFT_C2R")

    # Currently only support single dimension transform
    assert len(dim) == 1, "Only single dimension transform is supported"
    transform_dim = dim[0]

    # Normalize negative dimension
    ndim = input.ndim
    if transform_dim < 0:
        transform_dim = ndim + transform_dim

    # Get input shape info
    input_shape = list(input.shape)
    K = input_shape[transform_dim]  # Complex input size
    N = last_dim_size  # Output real size

    # Ensure input is contiguous
    input = input.contiguous()

    # Compute batch size M (product of all dims except transform dim)
    # and move transform dim to last position
    if transform_dim != ndim - 1:
        # Permute to move transform dim to last
        perm = list(range(ndim))
        perm.remove(transform_dim)
        perm.append(transform_dim)
        input = input.permute(perm)
        output_shape = [input_shape[i] for i in perm[:-1]] + [N]
        inverse_perm = [0] * ndim
        for i, p in enumerate(perm):
            inverse_perm[p] = i
    else:
        output_shape = input_shape[:-1] + [N]
        inverse_perm = None

    # Make contiguous after permute
    input = input.contiguous()

    # Compute batch size
    M = 1
    for i in range(ndim - 1):
        M *= input.shape[i]

    # Get real and imaginary parts
    # Complex tensor is stored as [..., 2] where last dim is [real, imag]
    input_real = torch.view_as_real(input)
    input_real_part = input_real[..., 0].contiguous()
    input_imag_part = input_real[..., 1].contiguous()

    # Reshape to 2D for kernel: [M, K]
    input_real_flat = input_real_part.view(M, K)
    input_imag_flat = input_imag_part.view(M, K)

    # Determine output dtype based on input complex dtype
    if input.dtype == torch.complex64:
        out_dtype = torch.float32
    elif input.dtype == torch.complex128:
        out_dtype = torch.float64
    else:
        out_dtype = torch.float32

    # Allocate output
    output_flat = torch.empty((M, N), dtype=out_dtype, device=input.device)

    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(input.device):
        _fft_c2r_kernel[grid](
            input_real_flat,
            input_imag_flat,
            output_flat,
            M,
            N,
            K,
            normalization,
        )

    # Reshape output
    output = output_flat.view(output_shape)

    # Permute back if needed
    if inverse_perm is not None:
        output = output.permute(inverse_perm)

    return output.contiguous()
