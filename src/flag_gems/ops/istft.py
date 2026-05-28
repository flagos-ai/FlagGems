"""
ISTFT (Inverse Short-Time Fourier Transform) operator implementation using Triton.

This module provides a high-performance implementation of the ISTFT operator
using Triton for GPU acceleration.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

PI = 3.141592653589793


def _log2(n: int) -> int:
    """Compute log2 of n (must be a power of 2)."""
    result = 0
    while n > 1:
        n >>= 1
        result += 1
    return result


def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _bitrev_indices(n: int, device: torch.device) -> torch.Tensor:
    """Generate bit-reversal indices for FFT."""
    log_n = _log2(n)
    result = torch.zeros(n, device=device, dtype=torch.int32)
    for i in range(n):
        rev = 0
        val = i
        for _ in range(log_n):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        result[i] = rev
    return result


@triton.jit
def overlap_add_kernel(
    ifft_out_real,
    ifft_out_imag,
    window,
    out_signal,
    n_frames: tl.constexpr,
    n_fft: tl.constexpr,
    hop_length: tl.constexpr,
    output_size: tl.constexpr,
    stride_ifft,
    stride_out,
    normalize: tl.constexpr,
):
    """Triton kernel for overlap-add operation in ISTFT."""
    pid = tl.program_id(0)
    frame_idx = pid

    if frame_idx >= n_frames:
        return

    offs = tl.arange(0, n_fft)
    mask = offs < n_fft

    win_vals = tl.load(window + offs, mask=mask, other=0.0)

    frame_base = frame_idx * stride_ifft
    ifft_real = tl.load(ifft_out_real + frame_base + offs, mask=mask, other=0.0)
    ifft_imag = tl.load(ifft_out_imag + frame_base + offs, mask=mask, other=0.0)

    ifft_real *= win_vals
    ifft_imag *= win_vals

    out_base = frame_idx * hop_length - n_fft // 2
    out_pos = out_base + offs

    out_mask = mask & (out_pos >= 0) & (out_pos < output_size)

    out_ptrs = out_signal + out_pos
    tl.atomic_add(out_ptrs, ifft_real, mask=out_mask)


def _prepare_istft_input(
    input: torch.Tensor, n_fft: int, onesided: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Prepare input tensor for ISTFT processing."""
    if input.is_complex():
        dtype = input.real.dtype
        real = input.real.contiguous()
        imag = input.imag.contiguous()
    else:
        dtype = input.dtype
        real = input.contiguous()
        imag = torch.zeros_like(real)

    if onesided and real.shape[-2] == n_fft // 2 + 1:
        n_freq = real.shape[-2]
        if n_freq != n_fft // 2 + 1:
            raise ValueError(
                f"onesided input has wrong shape: {real.shape[-2]} vs {n_fft // 2 + 1}"
            )

        batch_dims = real.shape[:-2]
        new_shape = batch_dims + (n_fft,) + real.shape[-1:]

        full_real = torch.zeros(new_shape, dtype=real.dtype, device=real.device)
        full_imag = torch.zeros(new_shape, dtype=imag.dtype, device=imag.device)

        full_real[..., :n_freq, :] = real
        full_imag[..., :n_freq, :] = imag

        if n_fft % 2 == 0:
            full_real[..., n_freq:, :] = real[..., 1 : n_freq - 1, :].flip(-2)
            full_imag[..., n_freq:, :] = -imag[..., 1 : n_freq - 1, :].flip(-2)
        else:
            full_real[..., n_freq:, :] = real[..., 1:n_freq, :].flip(-2)
            full_imag[..., n_freq:, :] = -imag[..., 1:n_freq, :].flip(-2)

        real = full_real
        imag = full_imag

    return real, imag, dtype


def _ifft_triton(
    real: torch.Tensor,
    imag: torch.Tensor,
    n_fft: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IFFT using the FFT kernel from fft.py.

    IFFT(x) = conj(FFT(conj(x))) / N
    """
    from flag_gems.ops.fft import (
        _bitrev_indices,
        _log2,
        _twiddle_tables,
        fft_kernel_triton,
    )

    n_total_frames = real.shape[0]

    imag_conj = -imag.contiguous()

    bitrev = _bitrev_indices(n_fft, real.device)
    tw_real, tw_imag = _twiddle_tables(n_fft, real.device)
    log_n = _log2(n_fft)

    buf0_real = torch.empty(
        (n_total_frames, n_fft), device=real.device, dtype=torch.float32
    )
    buf0_imag = torch.empty(
        (n_total_frames, n_fft), device=real.device, dtype=torch.float32
    )
    buf1_real = torch.empty(
        (n_total_frames, n_fft), device=real.device, dtype=torch.float32
    )
    buf1_imag = torch.empty(
        (n_total_frames, n_fft), device=real.device, dtype=torch.float32
    )

    grid = (n_total_frames,)
    fft_kernel_triton[grid](
        real,
        imag_conj,
        bitrev,
        tw_real,
        tw_imag,
        buf0_real,
        buf0_imag,
        buf1_real,
        buf1_imag,
        real.stride(0),
        buf0_real.stride(0),
        n_total_frames,
        N=n_fft,
        LOG_N=log_n,
        num_warps=4,
        num_stages=1,
    )

    total_swaps = (log_n + 1) // 2
    if total_swaps % 2 == 0:
        out_real = buf0_real
        out_imag = buf0_imag
    else:
        out_real = buf1_real
        out_imag = buf1_imag

    out_real /= n_fft
    out_imag = -out_imag / n_fft

    return out_real, out_imag


def istft(
    self: torch.Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex: bool = False,
) -> torch.Tensor:
    """
    Inverse Short Time Fourier Transform (ISTFT) using Triton.

    Args:
        self (Tensor): Input tensor containing the STFT result.
            Shape: (..., n_freq, n_frames)
        n_fft (int): Size of the FFT window. Must be a power of two.
        hop_length (int, optional): Number of samples between successive STFT frames.
            Defaults to `n_fft // 4`.
        win_length (int, optional): Size of the window function.
            Defaults to `n_fft`.
        window (Tensor, optional): Window function. Defaults to a Hann window.
        center (bool, optional): If True, the input signal is padded. Defaults to True.
        normalized (bool, optional): If True, normalize by window size. Defaults to False.
        onesided (bool, optional): If True, only positive frequencies are expected.
        length (int, optional): Output signal length.
        return_complex (bool, optional): If True, return complex tensor. Defaults to False.

    Returns:
        Tensor: The inverse STFT result.
    """
    pass

    if not _is_power_of_two(n_fft):
        raise ValueError(f"n_fft must be a power of two, got {n_fft}")

    if not self.is_cuda:
        return torch.istft(
            self,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
            length=length,
            return_complex=return_complex,
        )

    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if onesided is None:
        onesided = not self.is_complex()

    *batch_dims, n_freq, n_frames = self.shape

    real, imag, orig_dtype = _prepare_istft_input(self, n_fft, onesided)

    real = real.view(-1, n_fft, n_frames)
    imag = imag.view(-1, n_fft, n_frames)

    if window is None:
        window = torch.hann_window(win_length, device=self.device, dtype=torch.float32)
    else:
        window = window.to(torch.float32)

    if window.shape[0] < n_fft:
        padding = n_fft - window.shape[0]
        window = torch.nn.functional.pad(window, (0, padding))
    window = window.contiguous()

    n_total_frames = real.shape[0] * n_frames
    real_transposed = real.permute(0, 2, 1).contiguous().view(n_total_frames, n_fft)
    imag_transposed = imag.permute(0, 2, 1).contiguous().view(n_total_frames, n_fft)

    ifft_out_real, ifft_out_imag = _ifft_triton(real_transposed, imag_transposed, n_fft)

    if center:
        output_size = (n_frames - 1) * hop_length
    else:
        output_size = n_frames * hop_length + win_length - hop_length

    final_size = output_size

    if length is not None:
        final_size = length

    out_signal = torch.zeros(final_size, device=self.device, dtype=torch.float32)

    ifft_out_real = ifft_out_real.view(-1, n_frames, n_fft).contiguous()
    ifft_out_imag = ifft_out_imag.view(-1, n_frames, n_fft).contiguous()

    for batch_idx in range(ifft_out_real.shape[0]):
        batch_real = ifft_out_real[batch_idx]
        batch_imag = ifft_out_imag[batch_idx]

        grid_ola = (n_frames,)
        overlap_add_kernel[grid_ola](
            batch_real,
            batch_imag,
            window,
            out_signal,
            n_frames,
            n_fft,
            hop_length,
            output_size,
            batch_real.stride(0),
            out_signal.stride(0),
            normalize=normalized,
        )

    if length is not None and length != out_signal.shape[0]:
        out_signal = out_signal[:length]

    window_sums = torch.zeros(output_size, device=self.device, dtype=torch.float32)
    for frame_idx in range(n_frames):
        out_base = frame_idx * hop_length - n_fft // 2
        for i in range(n_fft):
            out_pos = out_base + i
            if out_pos >= 0 and out_pos < output_size:
                window_sums[out_pos] += window[i] ** 2

    out_signal[:output_size] /= window_sums.clamp(min=1e-9)

    if normalized:
        out_signal *= torch.sqrt(
            torch.tensor(n_fft, dtype=torch.float32, device=self.device)
        )

    if batch_dims:
        out_signal = out_signal.view(*batch_dims, -1)

    if orig_dtype in (torch.float16, torch.bfloat16):
        out_signal = out_signal.to(orig_dtype)

    if return_complex:
        return torch.complex(out_signal, torch.zeros_like(out_signal))

    return out_signal
