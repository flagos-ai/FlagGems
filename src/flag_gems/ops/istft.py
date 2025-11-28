import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ifft_onesided_kernel(
    spec_real_ptr,
    spec_imag_ptr,
    output_ptr,
    stride_batch,
    stride_freq,
    stride_frame,
    stride_output,
    scale,
    n_fft,
    n_frames,
    batch_size,
    BLOCK_T: tl.constexpr,
    N_FREQS: tl.constexpr,
):
    TWO_PI = 6.283185307179586

    pid = tl.program_id(0)
    batch_idx = pid // n_frames
    frame_idx = pid % n_frames

    if batch_idx >= batch_size:
        return

    time_pid = tl.program_id(1)
    time_idx = time_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = time_idx < n_fft
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    t_float = time_idx.to(tl.float32)
    n_fft_inv = 1.0 / n_fft
    base_offset = batch_idx * stride_batch + frame_idx * stride_frame

    freq_base = TWO_PI * n_fft_inv
    for freq in range(N_FREQS):
        offset = base_offset + freq * stride_freq
        real_val = tl.load(spec_real_ptr + offset)
        imag_val = tl.load(spec_imag_ptr + offset)

        angle = freq_base * freq * t_float
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)

        contrib = real_val * cos_val - imag_val * sin_val
        symmetry_factor = 2.0 if (freq > 0 and freq < N_FREQS - 1) else 1.0
        acc += contrib * symmetry_factor

    acc = acc * scale
    output_dtype = spec_real_ptr.dtype.element_ty
    acc = acc.to(output_dtype)

    output_offset = batch_idx * stride_output + frame_idx * n_fft
    tl.store(output_ptr + output_offset + time_idx, acc, mask=mask)


@triton.jit
def _ifft_full_kernel(
    spec_real_ptr,
    spec_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    stride_batch,
    stride_freq,
    stride_frame,
    stride_output,
    scale,
    n_fft,
    n_frames,
    batch_size,
    BLOCK_T: tl.constexpr,
    N_FREQS: tl.constexpr,
):
    TWO_PI = 6.283185307179586

    pid = tl.program_id(0)
    batch_idx = pid // n_frames
    frame_idx = pid % n_frames

    if batch_idx >= batch_size:
        return

    time_pid = tl.program_id(1)
    time_idx = time_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = time_idx < n_fft
    acc_real = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_T,), dtype=tl.float32)

    t_float = time_idx.to(tl.float32)
    n_fft_inv = 1.0 / n_fft
    base_offset = batch_idx * stride_batch + frame_idx * stride_frame

    freq_base = TWO_PI * n_fft_inv
    for freq in range(N_FREQS):
        offset = base_offset + freq * stride_freq
        real_val = tl.load(spec_real_ptr + offset)
        imag_val = tl.load(spec_imag_ptr + offset)

        angle = freq_base * freq * t_float
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)

        acc_real += real_val * cos_val - imag_val * sin_val
        acc_imag += real_val * sin_val + imag_val * cos_val

    acc_real = acc_real * scale
    acc_imag = acc_imag * scale
    output_dtype = spec_real_ptr.dtype.element_ty
    acc_real = acc_real.to(output_dtype)
    acc_imag = acc_imag.to(output_dtype)

    output_offset = batch_idx * stride_output + frame_idx * n_fft
    tl.store(output_real_ptr + output_offset + time_idx, acc_real, mask=mask)
    tl.store(output_imag_ptr + output_offset + time_idx, acc_imag, mask=mask)


@triton.jit
def _overlap_add_kernel(
    frame_buffer_real_ptr,
    frame_buffer_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    envelope_ptr,
    window_ptr,
    stride_batch,
    stride_frame_buffer,
    stride_output,
    hop_length,
    n_fft,
    win_length,
    full_length,
    n_frames,
    batch_size,
    BLOCK_T: tl.constexpr,
    APPLY_WINDOW: tl.constexpr,
    HAS_IMAG: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // n_frames
    frame_idx = pid % n_frames

    if batch_idx >= batch_size:
        return

    local_pid = tl.program_id(1)
    local_idx = local_pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = local_idx < win_length

    frame_offset = frame_idx * hop_length
    output_idx = frame_offset + local_idx
    output_mask = mask & (output_idx < full_length)

    buffer_offset = batch_idx * stride_frame_buffer + frame_idx * n_fft

    frame_real = tl.load(
        frame_buffer_real_ptr + buffer_offset + local_idx, mask=mask, other=0.0
    )
    frame_imag = (
        tl.load(frame_buffer_imag_ptr + buffer_offset + local_idx, mask=mask, other=0.0)
        if HAS_IMAG
        else 0.0
    )

    window_vals = (
        tl.load(window_ptr + local_idx, mask=mask, other=1.0) if APPLY_WINDOW else 1.0
    )

    if APPLY_WINDOW:
        frame_real = frame_real * window_vals
        if HAS_IMAG:
            frame_imag = frame_imag * window_vals

    output_offset = batch_idx * stride_output

    tl.atomic_add(
        output_real_ptr + output_offset + output_idx, frame_real, mask=output_mask
    )
    if HAS_IMAG:
        tl.atomic_add(
            output_imag_ptr + output_offset + output_idx, frame_imag, mask=output_mask
        )

    if APPLY_WINDOW:
        tl.atomic_add(
            envelope_ptr + output_offset + output_idx, window_vals, mask=output_mask
        )


@triton.jit
def _normalize_kernel(
    output_real_ptr,
    output_imag_ptr,
    envelope_ptr,
    stride_batch,
    length,
    batch_size,
    BLOCK_T: tl.constexpr,
    HAS_IMAG: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    if batch_idx >= batch_size:
        return

    pid = tl.program_id(1)
    idx = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = idx < length

    batch_offset = batch_idx * stride_batch

    output_real = tl.load(output_real_ptr + batch_offset + idx, mask=mask)
    output_imag = (
        tl.load(output_imag_ptr + batch_offset + idx, mask=mask) if HAS_IMAG else 0.0
    )
    envelope = tl.load(envelope_ptr + batch_offset + idx, mask=mask)

    ENVELOPE_EPS = 1e-8
    envelope_safe = tl.where(envelope > ENVELOPE_EPS, envelope, 1.0)
    envelope_inv = 1.0 / envelope_safe

    output_real = output_real * envelope_inv
    if HAS_IMAG:
        output_imag = output_imag * envelope_inv

    tl.store(output_real_ptr + batch_offset + idx, output_real, mask=mask)
    if HAS_IMAG:
        tl.store(output_imag_ptr + batch_offset + idx, output_imag, mask=mask)


def istft(
    input_tensor,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=True,
    length=None,
    return_complex=False,
):
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if win_length <= 0 or win_length > n_fft:
        raise ValueError("win_length must be in (0, n_fft]")

    if input_tensor.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")

    if input_tensor.is_complex():
        spectrum = input_tensor
    else:
        if input_tensor.shape[-1] != 2:
            raise TypeError("Input tensor must be complex or have last dimension == 2")
        spectrum = torch.view_as_complex(input_tensor)

    spectrum = spectrum.contiguous()
    *batch_shape, n_freqs, n_frames = spectrum.shape
    batch_size = math.prod(batch_shape) if batch_shape else 1

    expected_freqs = n_fft // 2 + 1 if onesided else n_fft
    if n_freqs != expected_freqs:
        raise ValueError(
            f"Expected frequency dimension {expected_freqs} but got {n_freqs}"
        )

    device = spectrum.device
    real_dtype = spectrum.real.dtype

    use_intermediate_precision = real_dtype == torch.float16
    if use_intermediate_precision:
        spectrum = torch.complex(
            spectrum.real.to(torch.float32), spectrum.imag.to(torch.float32)
        )
        real_dtype = torch.float32

    if window is None:
        window = torch.ones(win_length, device=device, dtype=real_dtype)
    else:
        if window.numel() != win_length:
            raise ValueError("window length mismatch")
        window = window.to(device=device, dtype=real_dtype)

    scale = 1.0 / math.sqrt(n_fft) if normalized else 1.0 / n_fft

    spectrum = spectrum.reshape(batch_size, n_freqs, n_frames)
    spec_real = spectrum.real.contiguous()
    spec_imag = spectrum.imag.contiguous()

    stride_freq = spec_real.stride(-2)
    stride_frame = spec_real.stride(-1)

    if center:
        pad = n_fft // 2
        full_length = n_frames * hop_length + n_fft - hop_length
    else:
        pad = 0
        full_length = n_frames * hop_length + n_fft - hop_length

    if length is not None:
        full_length = length

    needs_imag = not onesided or return_complex

    output_real = torch.zeros(
        (batch_size, full_length), dtype=real_dtype, device=device
    )
    output_imag = (
        torch.zeros((batch_size, full_length), dtype=real_dtype, device=device)
        if needs_imag
        else None
    )
    envelope = torch.zeros((batch_size, full_length), dtype=real_dtype, device=device)

    ifft_buffer_real = torch.empty(
        (batch_size, n_frames * n_fft), dtype=real_dtype, device=device
    )
    ifft_buffer_imag = (
        torch.empty((batch_size, n_frames * n_fft), dtype=real_dtype, device=device)
        if needs_imag
        else ifft_buffer_real
    )

    stride_batch = spec_real.stride(0)
    stride_output = ifft_buffer_real.stride(0)

    grid_ifft = lambda meta: (
        batch_size * n_frames,
        triton.cdiv(n_fft, meta["BLOCK_T"]),
    )

    if onesided:
        _ifft_onesided_kernel[grid_ifft](
            spec_real,
            spec_imag,
            ifft_buffer_real,
            stride_batch,
            stride_freq,
            stride_frame,
            stride_output,
            scale,
            n_fft,
            n_frames,
            batch_size,
            BLOCK_T=256,
            N_FREQS=n_freqs,
        )
    else:
        _ifft_full_kernel[grid_ifft](
            spec_real,
            spec_imag,
            ifft_buffer_real,
            ifft_buffer_imag,
            stride_batch,
            stride_freq,
            stride_frame,
            stride_output,
            scale,
            n_fft,
            n_frames,
            batch_size,
            BLOCK_T=256,
            N_FREQS=n_freqs,
        )

    grid_overlap = lambda meta: (
        batch_size * n_frames,
        triton.cdiv(win_length, meta["BLOCK_T"]),
    )
    stride_frame_buffer = ifft_buffer_real.stride(0)
    stride_output_batch = output_real.stride(0)

    _overlap_add_kernel[grid_overlap](
        ifft_buffer_real,
        ifft_buffer_imag if needs_imag else ifft_buffer_real,
        output_real,
        output_imag if needs_imag else output_real,
        envelope,
        window,
        stride_batch,
        stride_frame_buffer,
        stride_output_batch,
        hop_length,
        n_fft,
        win_length,
        full_length,
        n_frames,
        batch_size,
        BLOCK_T=256,
        APPLY_WINDOW=True,
        HAS_IMAG=needs_imag,
    )

    grid_norm = lambda meta: (batch_size, triton.cdiv(full_length, meta["BLOCK_T"]))
    _normalize_kernel[grid_norm](
        output_real,
        output_imag if needs_imag else output_real,
        envelope,
        stride_output_batch,
        full_length,
        batch_size,
        BLOCK_T=256,
        HAS_IMAG=needs_imag,
    )

    if pad:
        output_real = output_real[..., pad:-pad]
        if needs_imag:
            output_imag = output_imag[..., pad:-pad]

    if batch_size == 1:
        output_real = output_real.squeeze(0)
        if needs_imag:
            output_imag = output_imag.squeeze(0)

    if use_intermediate_precision:
        output_real = output_real.to(torch.float16)
        if needs_imag:
            output_imag = output_imag.to(torch.float16)

    if return_complex:
        if needs_imag:
            return torch.complex(output_real, output_imag)
        else:
            return output_real
    else:
        return output_real
