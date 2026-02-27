import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _get_normalization(norm: Optional[str]) -> int:
    """
    Convert norm string to integer for _fft_r2c.

    normalization values:
    - 0: backward (no normalization)
    - 1: ortho (1/sqrt(n) normalization)
    - 2: forward (1/n normalization)
    """
    if norm is None or norm == "backward":
        return 0
    elif norm == "ortho":
        return 1
    elif norm == "forward":
        return 2
    else:
        raise ValueError(f"Invalid norm mode: {norm}")


def fft_rfft(
    input: torch.Tensor,
    n: Optional[int] = None,
    dim: int = -1,
    norm: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the one-dimensional Fourier transform of real-valued input.

    This FlagGems implementation uses the low-level _fft_r2c operation
    which uses highly optimized cuFFT library. FFT is a fundamentally
    different algorithm (Cooley-Tukey) that doesn't fit the pointwise
    or reduction patterns common in Triton kernels.

    Args:
        input: Real input tensor
        n: Signal length. If given, input is zero-padded or trimmed to this length.
        dim: Dimension along which to take the FFT
        norm: Normalization mode ("forward", "backward", or "ortho")

    Returns:
        Complex tensor containing the FFT result with shape (..., n//2+1)
    """
    logger.debug("GEMS FFT_RFFT")

    # Validate input
    if input.is_complex():
        raise RuntimeError("rfft expects a real input tensor, but got complex")

    # Handle dimension
    ndim = input.ndim
    if ndim == 0:
        raise RuntimeError("rfft: input must have at least 1 dimension")

    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim-1}], but got {dim})"
        )

    # Normalize dim
    if dim < 0:
        dim = dim + ndim

    # Handle n parameter (signal length)
    signal_length = input.shape[dim]
    if n is None:
        n = signal_length

    # Pad or truncate if needed
    if n != signal_length:
        if n > signal_length:
            # Zero-pad
            pad_shape = list(input.shape)
            pad_shape[dim] = n - signal_length
            padding = torch.zeros(
                pad_shape, dtype=input.dtype, device=input.device
            )
            input = torch.cat([input, padding], dim=dim)
        else:
            # Truncate
            input = torch.narrow(input, dim, 0, n)

    # Get normalization integer
    normalization = _get_normalization(norm)

    # Call the low-level _fft_r2c operation
    # This avoids recursion as it's not the registered fft_rfft op
    return torch.ops.aten._fft_r2c.default(input, [dim], normalization, True)
