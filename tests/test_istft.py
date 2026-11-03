# Copyright 2026 FlagOS Contributors
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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# torch.istft requires a complex spectrogram (the return_complex=True output of
# torch.stft). cuFFT rejects bfloat16 spectrograms, so bfloat16 is not
# reachable through a stft/istft round trip; float16 spectrograms are complex32
# and torch.istft accepts them directly.
FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64]

# torch computes the fp16/complex32 inverse transform in half precision, so its
# reference carries ~1e-3 relative error; the Triton kernels accumulate in fp32
# and are therefore closer to the true signal. The tolerance follows the
# reference's own precision rather than the input dtype.
_ATOL = {
    torch.float16: 5e-3,
    torch.float32: 1e-4,
    torch.float64: 1e-7,
}

# (n_fft, hop_length, win_length) combinations: the default configuration, a
# minimal transform, a window shorter than the transform size, and a large
# transform.
FFT_PARAMS = [
    (256, 64, 256),
    (16, 4, 16),
    (512, 128, 256),
    (1024, 256, 1024),
]


def _stft(signal, n_fft, hop_length, win_length, window, **kwargs):
    return torch.stft(
        signal,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        **kwargs,
    )


def _build_case(
    signal_len, batch, n_fft, hop_length, win_length, dtype, rect_window=False, **kwargs
):
    """Return (gems spec, gems window, reference spec, reference window).

    The same underlying signal and window drive both sides: the reference
    operands are the upcast CPU copies (utils.to_reference), so the two
    spectrograms describe one signal.
    """
    device = flag_gems.device
    if rect_window:
        window = torch.ones(win_length, dtype=dtype, device=device)
    else:
        window = torch.hann_window(win_length, dtype=dtype, device=device)
    shape = (signal_len,) if batch == 0 else (batch, signal_len)
    signal = torch.randn(shape, dtype=dtype, device=device)
    spec = _stft(signal, n_fft, hop_length, win_length, window, **kwargs)

    ref_signal = utils.to_reference(signal, upcast=True)
    ref_window = utils.to_reference(window, upcast=True)
    ref_spec = _stft(ref_signal, n_fft, hop_length, win_length, ref_window, **kwargs)
    return spec, window, ref_spec, ref_window


def _assert_close(res, ref, dtype, atol):
    utils.gems_assert_close(utils.to_cpu(res, ref), ref, dtype, atol=atol)


@pytest.mark.istft
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n_fft,hop_length,win_length", FFT_PARAMS)
def test_istft_roundtrip(dtype, n_fft, hop_length, win_length):
    signal_len = 4096
    batch = 2
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len,
    )
    ref_out = torch.istft(
        ref_spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=ref_window,
        length=signal_len,
    ).to(dtype)

    assert res_out.shape == ref_out.shape
    _assert_close(res_out, ref_out, dtype, _ATOL[dtype])


@pytest.mark.istft
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_istft_1d_signal(dtype):
    signal_len = 1024
    n_fft, hop_length, win_length = 256, 64, 256
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, 0, n_fft, hop_length, win_length, dtype
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len,
    )
    ref_out = torch.istft(
        ref_spec, n_fft, hop_length, win_length, ref_window, length=signal_len
    ).to(dtype)

    assert res_out.dim() == 1 and ref_out.dim() == 1
    _assert_close(res_out, ref_out, dtype, _ATOL[dtype])


@pytest.mark.istft
# fp16 is omitted here: the complex-return and normalized variants are covered
# by test_istft_roundtrip for all three dtypes, and the fp64 path here already
# pins the exact-arithmetic behavior this test targets.
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_istft_variants(dtype):
    signal_len = 2048
    n_fft, hop_length, win_length = 256, 64, 256
    batch = 2
    atol = _ATOL[dtype]

    # normalized=True round trip
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype, normalized=True
    )
    res = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        length=signal_len,
    )
    ref = torch.istft(
        ref_spec,
        n_fft,
        hop_length,
        win_length,
        ref_window,
        normalized=True,
        length=signal_len,
    ).to(dtype)
    _assert_close(res, ref, dtype, atol)

    # explicit length shorter than the expected output
    res = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len - 8,
    )
    ref = torch.istft(
        ref_spec, n_fft, hop_length, win_length, ref_window, length=signal_len - 8
    ).to(dtype)
    _assert_close(res, ref, dtype, atol)

    # no explicit length (returns the full centered signal)
    res = flag_gems.istft(
        spec, n_fft, hop_length=hop_length, win_length=win_length, window=window
    )
    ref = torch.istft(ref_spec, n_fft, hop_length, win_length, ref_window).to(dtype)
    _assert_close(res, ref, dtype, atol)

    # two-sided spectrogram (onesided=False), inferred onesided=None
    spec2, window2, ref_spec2, ref_window2 = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype, onesided=False
    )
    res2 = flag_gems.istft(
        spec2,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window2,
        length=signal_len,
    )
    ref2 = torch.istft(
        ref_spec2, n_fft, hop_length, win_length, ref_window2, length=signal_len
    ).to(dtype)
    _assert_close(res2, ref2, dtype, atol)

    # return_complex=True on a two-sided spectrogram
    res3 = flag_gems.istft(
        spec2,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window2,
        return_complex=True,
        onesided=False,
    )
    ref3 = torch.istft(
        ref_spec2,
        n_fft,
        hop_length,
        win_length,
        ref_window2,
        return_complex=True,
        onesided=False,
    )
    assert res3.is_complex() and ref3.is_complex()
    for res_c, ref_c in (
        (res3.real, ref3.real.to(dtype)),
        (res3.imag, ref3.imag.to(dtype)),
    ):
        _assert_close(res_c, ref_c, dtype, atol)


@pytest.mark.istft
def test_istft_center_false():
    # With center=False the hann-window overlap-add envelope vanishes at the
    # tail and torch rejects the call; a rectangular window keeps the envelope
    # non-zero on both sides.
    dtype = torch.float32
    signal_len = 2048
    n_fft, hop_length, win_length = 256, 64, 256
    length = signal_len - n_fft
    spec, window, ref_spec, ref_window = _build_case(
        signal_len,
        2,
        n_fft,
        hop_length,
        win_length,
        dtype,
        rect_window=True,
        center=False,
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=length,
    )
    ref_out = torch.istft(
        ref_spec,
        n_fft,
        hop_length,
        win_length,
        ref_window,
        center=False,
        length=length,
    ).to(dtype)
    _assert_close(res_out, ref_out, dtype, _ATOL[dtype])


@pytest.mark.istft
def test_istft_errors():
    device = flag_gems.device
    n_fft, hop_length, win_length = 256, 64, 256
    window = torch.hann_window(win_length, device=device)
    signal = torch.randn(1024, device=device)
    spec = torch.stft(
        signal, n_fft, hop_length, win_length, window, return_complex=True
    )
    spec2 = torch.stft(
        signal,
        n_fft,
        hop_length,
        win_length,
        window,
        return_complex=True,
        onesided=False,
    )

    # real input is rejected
    real_spec = torch.view_as_real(spec)
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            real_spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
    # onesided output with return_complex=True is rejected
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
    # frequency dimension must match n_fft/2+1 (onesided) or n_fft (two-sided)
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            onesided=False,
        )
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec2,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            onesided=True,
        )
    # hop_length must not exceed win_length
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec, n_fft, hop_length=n_fft + 1, win_length=win_length, window=window
        )
    # win_length must not exceed n_fft
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec, n_fft, hop_length=hop_length, win_length=n_fft + 1, window=window
        )
    # window shape must match win_length
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window[: win_length - 1],
        )
    # empty input is rejected
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            torch.empty(0, n_fft // 2 + 1, 4, dtype=torch.complex64, device=device),
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
