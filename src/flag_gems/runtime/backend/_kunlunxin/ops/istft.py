# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# Output-major overlap-add (atomic-free).
#
# The generic implementation scatters each (channel, frame) block into the
# output signal with ``tl.atomic_add``.  On this XPU backend cross-program
# atomics are unreliable (masked ``tl.atomic_add`` silently drops ~1% of the
# updates), which corrupts the reconstructed signal.  Instead each program
# owns one block of output samples for one channel and serially accumulates the
# contributions of the (few) frames covering that block.  No cross-program
# atomic is ever issued.
#
# The window is folded into the frames on the host before the kernel runs
# (``frames = frames * window``), so the inner loop is a single masked load and
# an add with no per-iteration multiply.  The load uses the affine address
# ``local = offs - fi * hop`` (not a clamped one) so the compiler emits
# coalesced vector loads; out-of-range lanes are masked and then zeroed by
# ``tl.where`` (the ``other`` value of a masked load is ignored on this
# backend, hence the explicit select).
@libentry()
@triton.jit
def istft_overlap_add_kernel(
    frames_ptr,  # (channel, n_frames, n_fft) real frames, window already applied
    signal_ptr,  # (channel, out_len) accumulated output signal
    n_frames,
    n_fft,
    hop_length,
    signal_stride0,
    out_len,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    channel = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_len

    acc = tl.zeros([BLOCK], dtype=frames_ptr.dtype.element_ty)

    # Frame f covers padded samples [f*hop, f*hop + n_fft).  The frame range
    # overlapping this output block is a pure function of the block index, so
    # compute it with scalar arithmetic (no cross-lane reductions, which are
    # expensive on this backend).  The in-frame mask below discards any lanes
    # outside the true overlap, so a slightly wide bound is harmless.
    block_start = pid * BLOCK
    f_start = tl.maximum((block_start - n_fft + hop_length) // hop_length, 0)
    f_end = tl.minimum((block_start + BLOCK - 1) // hop_length + 1, n_frames)

    frame_base = channel * n_frames * n_fft
    for fi in range(f_start, f_end):
        local = offs - fi * hop_length
        in_frame = (local >= 0) & (local < n_fft) & mask
        frame = tl.load(
            frames_ptr + frame_base + fi * n_fft + local, mask=in_frame, other=0.0
        )
        acc += tl.where(in_frame, frame, 0.0)

    tl.store(signal_ptr + channel * signal_stride0 + offs, acc, mask=mask)


@libentry()
@triton.jit
def istft_normalize_kernel(
    signal_ptr,
    envelope_ptr,
    n,
    eps,
    stride0,
    BLOCK: tl.constexpr,
):
    # signal_ptr is a (channel, n) sliced view; the envelope is contiguous.
    pid = tl.program_id(0)
    ch = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = ch * stride0
    y = tl.load(signal_ptr + base + offs, mask=mask, other=0.0)
    w = tl.load(envelope_ptr + offs, mask=mask, other=1.0)
    y = y / w
    tl.store(signal_ptr + base + offs, y, mask=mask)


def istft(
    input,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=None,
    length=None,
    return_complex=False,
):
    logger.debug("GEMS_KUNLUNXIN ISTFT")
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if not input.is_complex():
        raise RuntimeError(
            "istft requires a complex-valued input tensor matching the "
            "output from stft with return_complex=True."
        )

    if input.dtype not in (torch.complex32, torch.complex64, torch.complex128):
        raise TypeError(
            "istft supports only complex32, complex64 and complex128 inputs, "
            f"got {input.dtype}"
        )

    inp = torch.view_as_real(input.resolve_conj())
    input_dim = inp.dim()
    n_frames = inp.shape[-2]
    fft_size = inp.shape[-3]
    if fft_size * 0 == 0 and inp.numel() == 0:
        raise RuntimeError("istft: input tensor cannot be empty.")
    if input_dim not in (3, 4):
        raise RuntimeError(
            f"istft: expected a tensor with 3 or 4 dimensions, but got {input_dim}"
        )
    if inp.shape[-1] != 2:
        raise RuntimeError(
            f"istft: expected the last dimension to be 2, but got {inp.shape[-1]}"
        )

    if onesided is None:
        onesided = fft_size != n_fft
    if onesided:
        if n_fft // 2 + 1 != fft_size:
            raise RuntimeError(
                f"istft: expected the frequency dimension to match n_fft / 2 + 1 "
                f"when onesided=True, but got {fft_size}"
            )
    else:
        if n_fft != fft_size:
            raise RuntimeError(
                f"istft: expected the frequency dimension to match n_fft when "
                f"onesided=False, but got {fft_size}"
            )
    if not (0 < hop_length <= win_length):
        raise RuntimeError("istft: expected 0 < hop_length <= win_length")
    if not (0 < win_length <= n_fft):
        raise RuntimeError("istft: expected 0 < win_length <= n_fft")
    if window is not None and (window.dim() != 1 or window.shape[0] != win_length):
        raise RuntimeError(
            "istft: Invalid window shape. window has to be 1D and length of "
            "`win_length`"
        )

    # Frame data as complex (channel, n_frames, fft_size); cuFFT requires
    # contiguous input.
    frames = input.resolve_conj().transpose(-1, -2)  # (..., n_frames, fft_size)
    if input_dim == 3:
        frames = frames.unsqueeze(0)
    channel = frames.shape[0]
    frames = frames.contiguous()
    if frames.dtype == torch.complex32:
        # torch_xmlir lacks a complex32 irfft/ifft ("kcomplex32 combined is
        # unsupported"); upcast to complex64 so the transform runs on-device and
        # the kernels below accumulate in fp32 (closer to the true signal).
        frames = frames.to(torch.complex64)

    # torch_xmlir's irfft/ifft mis-handle the "ortho" norm (it applies an extra
    # 1/n_fft scaling on this XPU backend), so the normalized transform is
    # computed via "backward" plus a manual sqrt(n_fft) scale.
    norm = "backward"
    scale = float(n_fft) ** 0.5 if normalized else 1.0
    if return_complex:
        if onesided:
            raise RuntimeError(
                "istft: cannot have onesided output if window or input is complex"
            )
        frames_t = torch.fft.ifft(frames, n=n_fft, dim=-1, norm=norm)
        frames_r = frames_t * scale
        real_out = False
    else:
        if not onesided:
            frames = frames[..., : n_fft // 2 + 1]
        # cuFFT c2r writes a real (channel, n_frames, n_fft) tensor.
        frames_r = torch.fft.irfft(frames, n=n_fft, dim=-1, norm=norm) * scale
        real_out = True

    # Real component dtype of the complex spectrum (complex32 -> float16,
    # complex64 -> float32, complex128 -> float64).  The signal/envelope are
    # computed in ``frames_r.dtype`` (fp32 after the complex32 upcast) and the
    # result is downcast to ``out_real_dtype`` at the end so complex32 inputs
    # yield a float16 output like ``torch.istft``.
    _real_of_complex = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }
    out_real_dtype = _real_of_complex.get(input.dtype, torch.float32)
    # Window/envelope are always real; for return_complex=True ``frames_r`` is
    # complex (ifft output), so take its real component dtype.
    compute_dtype = frames_r.real.dtype

    # Window: mirror torch's centering of the window into the n_fft frame.
    if window is None:
        window_full = torch.ones(n_fft, dtype=compute_dtype, device=frames_r.device)
    elif win_length != n_fft:
        left = (n_fft - win_length) // 2
        window_full = torch.constant_pad_nd(
            window, (left, n_fft - win_length - left), 0.0
        ).to(compute_dtype)
    else:
        window_full = window.to(compute_dtype)

    expected_len = n_fft + hop_length * (n_frames - 1)
    BLOCK = 1024

    # Fold the window into the frames once on the host so the overlap-add inner
    # loop is a single load + add (no multiply), and precompute the squared
    # window for the envelope.
    window_full = window_full.contiguous()
    window_sq = (window_full * window_full).contiguous()
    # The envelope is the overlap-add of the squared window.  Reuse the
    # overlap-add kernel with the squared window replicated over the frame axis
    # (a (n_frames, n_fft) tensor laid out like the frames).  Loading from a
    # small (n_fft,) tensor in the kernel compiles to slow gather code on this
    # backend, whereas the replicated frame-shaped tensor uses the same fast
    # coalesced path as the signal.
    window_sq_rep = window_sq.unsqueeze(0).expand(n_frames, n_fft).contiguous()

    def _overlap_add(frame_planes):
        # frame_planes: list of real (channel, n_frames, n_fft) tensors; one
        # output signal buffer per plane.
        signals = []
        for plane in frame_planes:
            plane_w = (plane * window_full).contiguous()
            sig = torch.zeros(
                channel, expected_len, dtype=compute_dtype, device=frames_r.device
            )
            grid = (triton.cdiv(expected_len, BLOCK), channel)
            istft_overlap_add_kernel[grid](
                plane_w,
                sig,
                n_frames,
                n_fft,
                hop_length,
                sig.stride(0),
                expected_len,
                BLOCK=BLOCK,
            )
            signals.append(sig)
        return signals

    if real_out:
        signal = _overlap_add([frames_r])[0]
        signal_r = signal_i = None
    else:
        # Complex output: overlap-add the real and imaginary planes separately
        # against the same (real) window.
        signal_r, signal_i = _overlap_add([frames_r.real, frames_r.imag])
        signal = None

    # Envelope from the squared window (same overlap-add, deterministic).
    envelope = torch.zeros(expected_len, dtype=compute_dtype, device=frames_r.device)
    env_grid = (triton.cdiv(expected_len, BLOCK), 1)
    istft_overlap_add_kernel[env_grid](
        window_sq_rep,
        envelope,
        n_frames,
        n_fft,
        hop_length,
        envelope.stride(0),
        expected_len,
        BLOCK=BLOCK,
    )

    # Trim center padding / requested length on both signal and envelope.
    start = n_fft // 2 if center else 0
    if length is not None:
        end = start + length
    elif center:
        end = expected_len - n_fft // 2
    else:
        end = expected_len

    w_env = envelope[start:end].contiguous()

    def _normalize(plane):
        n_out = plane.shape[1]
        if n_out > 0:
            BLOCK_N = 1024
            norm_grid = (triton.cdiv(n_out, BLOCK_N), plane.shape[0])
            istft_normalize_kernel[norm_grid](
                plane, w_env, n_out, 1e-11, plane.stride(0), BLOCK=BLOCK_N
            )
        return plane

    if real_out:
        y = _normalize(signal[:, start:end])
    else:
        yr = _normalize(signal_r[:, start:end])
        yi = _normalize(signal_i[:, start:end])
        y = torch.complex(yr, yi)

    if end > expected_len:
        pad = end - expected_len
        y = torch.constant_pad_nd(y, (0, pad), 0.0)
    if input_dim == 3:
        y = y.squeeze(0)

    # Downcast to torch's real dtype for complex32 inputs.
    if real_out:
        if y.dtype != out_real_dtype:
            y = y.to(out_real_dtype)
    else:
        if y.real.dtype != out_real_dtype:
            y = torch.complex(y.real.to(out_real_dtype), y.imag.to(out_real_dtype))
    return y
