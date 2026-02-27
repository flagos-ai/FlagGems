import logging
from typing import List, Optional, Sequence, Union

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# FFT normalization modes mapping for _fft_c2c internal function
# Based on PyTorch internals: 0=backward (none), 1=ortho (1/sqrt(n)), 2=forward (1/n)
NORM_MODES = {
    None: 0,  # backward (no normalization)
    "backward": 0,
    "ortho": 1,  # normalize by 1/sqrt(n)
    "forward": 2,  # normalize by 1/n
}


@libentry()
@triton.jit
def _complex_copy_kernel(
    in_real_ptr,
    in_imag_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy real/imag parts to complex output tensor."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    real_val = tl.load(in_real_ptr + offsets, mask=mask, other=0.0)
    imag_val = tl.load(in_imag_ptr + offsets, mask=mask, other=0.0)

    # Store interleaved (real, imag) pairs
    tl.store(out_ptr + offsets * 2, real_val, mask=mask)
    tl.store(out_ptr + offsets * 2 + 1, imag_val, mask=mask)


def _normalize_dims(ndim: int, dim: Optional[Sequence[int]]) -> List[int]:
    """Normalize dimension list, handling None and negative indices."""
    if dim is None:
        return list(range(ndim))
    dims = list(dim) if not isinstance(dim, int) else [dim]
    return [(d % ndim) for d in dims]


def _get_fft_size(
    input_shape: torch.Size,
    s: Optional[Sequence[int]],
    dim: List[int],
) -> List[int]:
    """Compute the FFT sizes for each dimension."""
    if s is None:
        return [input_shape[d] for d in dim]
    s_list = list(s)
    result = []
    for i, d in enumerate(dim):
        if i < len(s_list) and s_list[i] != -1:
            result.append(s_list[i])
        else:
            result.append(input_shape[d])
    return result


def fft_fftn(
    self: torch.Tensor,
    s: Optional[Sequence[int]] = None,
    dim: Optional[Sequence[int]] = None,
    norm: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute the N-dimensional discrete Fourier transform.

    This implementation provides a FlagGems-compatible interface for FFT operations.
    It handles input preprocessing and delegates to optimized FFT implementations.

    Args:
        self: Input tensor (real or complex)
        s: Signal size in the transformed dimensions. If given, each dimension
           dim[i] will either be zero-padded or trimmed to the length s[i].
        dim: Dimensions to be transformed. Default: all dimensions.
        norm: Normalization mode. One of "forward", "backward", or "ortho".
              Default: "backward" (no normalization).

    Returns:
        Complex tensor containing the FFT result.
    """
    logger.debug("GEMS FFT_FFTN")

    # Input validation
    if self.numel() == 0:
        # Handle empty tensor
        return torch.empty(
            self.shape,
            dtype=torch.complex64 if self.dtype in (torch.float32, torch.float16, torch.bfloat16)
            else torch.complex128 if self.dtype == torch.float64
            else self.dtype,
            device=self.device,
        )

    # Normalize dimensions
    ndim = self.ndim
    dims = _normalize_dims(ndim, dim)

    # Get FFT sizes
    fft_sizes = _get_fft_size(self.shape, s, dims)

    # Validate normalization mode
    if norm is not None and norm not in NORM_MODES:
        raise ValueError(f"Invalid normalization mode: {norm}. Must be one of {list(NORM_MODES.keys())}")

    norm_mode = NORM_MODES.get(norm, 0)

    # Handle input tensor preparation
    x = self.contiguous()

    # For real input, we need to handle the real-to-complex conversion
    is_real_input = not x.is_complex()

    with torch_device_fn.device(x.device):
        if is_real_input:
            # Real-to-complex FFT path
            # Use torch's internal _fft_r2c for the first dimension, then _fft_c2c for rest
            # Note: _fft_r2c returns the one-sided FFT, we need full FFT for fftn

            # First, convert real input to complex
            if x.dtype == torch.float64:
                complex_dtype = torch.complex128
            elif x.dtype in (torch.float32, torch.float16, torch.bfloat16):
                complex_dtype = torch.complex64
            else:
                # For integer types, convert to float first
                x = x.to(torch.float32)
                complex_dtype = torch.complex64

            # Prepare the input with correct sizes (padding/truncation)
            x_prepared = x
            for i, d in enumerate(dims):
                target_size = fft_sizes[i]
                current_size = x_prepared.shape[d]
                if target_size != current_size:
                    if target_size > current_size:
                        # Pad with zeros
                        pad_size = target_size - current_size
                        pad_shape = list(x_prepared.shape)
                        pad_shape[d] = pad_size
                        padding = torch.zeros(pad_shape, dtype=x_prepared.dtype, device=x_prepared.device)
                        x_prepared = torch.cat([x_prepared, padding], dim=d)
                    else:
                        # Truncate
                        x_prepared = x_prepared.narrow(d, 0, target_size)

            # Convert to complex
            x_complex = torch.complex(x_prepared.to(torch.float32 if complex_dtype == torch.complex64 else torch.float64),
                                      torch.zeros_like(x_prepared, dtype=torch.float32 if complex_dtype == torch.complex64 else torch.float64))

            # Perform complex-to-complex FFT using internal aten op
            result = torch.ops.aten._fft_c2c(x_complex, dims, norm_mode, True)  # forward=True

        else:
            # Complex-to-complex FFT path
            x_prepared = x
            for i, d in enumerate(dims):
                target_size = fft_sizes[i]
                current_size = x_prepared.shape[d]
                if target_size != current_size:
                    if target_size > current_size:
                        # Pad with zeros
                        pad_size = target_size - current_size
                        pad_shape = list(x_prepared.shape)
                        pad_shape[d] = pad_size
                        padding = torch.zeros(pad_shape, dtype=x_prepared.dtype, device=x_prepared.device)
                        x_prepared = torch.cat([x_prepared, padding], dim=d)
                    else:
                        # Truncate
                        x_prepared = x_prepared.narrow(d, 0, target_size)

            # Perform complex-to-complex FFT
            result = torch.ops.aten._fft_c2c(x_prepared, dims, norm_mode, True)  # forward=True

    return result
