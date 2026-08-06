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

# The Triton FFT implementation supports complex64 input and float32 output;
# these dtypes are fixed and independent of the (mode-dependent) dtype lists in
# accuracy_utils, so they are hard-coded rather than filtered from those lists.
ISTFT_COMPLEX_DTYPE = torch.complex64
ISTFT_REAL_DTYPE = torch.float32


def _complex_randn(shape):
    real = torch.randn(shape, device=flag_gems.device, dtype=ISTFT_REAL_DTYPE)
    imag = torch.randn_like(real)
    return torch.complex(real, imag).to(ISTFT_COMPLEX_DTYPE)


@pytest.mark.istft
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("n_fft,hop_length,n_frames", [(64, 16, 8), (256, 64, 12)])
def test_istft_onesided(batched, normalized, n_fft, hop_length, n_frames):
    n_freq = n_fft // 2 + 1
    shape = (2, n_freq, n_frames) if batched else (n_freq, n_frames)
    inp = _complex_randn(shape)
    window = torch.hann_window(n_fft, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_window = utils.to_reference(window)
    expected = torch.istft(
        ref_inp,
        n_fft,
        hop_length=hop_length,
        window=ref_window,
        normalized=normalized,
    )

    with flag_gems.use_gems():
        actual = torch.istft(
            inp,
            n_fft,
            hop_length=hop_length,
            window=window,
            normalized=normalized,
        )

    utils.gems_assert_close(actual, expected, torch.float32, reduce_dim=n_fft)


@pytest.mark.istft
@pytest.mark.parametrize(
    "center,win_length,length",
    [
        (True, 96, None),
        (False, 128, None),
        (True, 128, 300),
    ],
)
def test_istft_window_padding_center_and_length(center, win_length, length):
    n_fft = 128
    inp = _complex_randn((2, 65, 8))
    if center and win_length < n_fft:
        window = torch.hamming_window(win_length, device=flag_gems.device)
    else:
        window = torch.ones(win_length, device=flag_gems.device)
    kwargs = {
        "n_fft": n_fft,
        "hop_length": 32,
        "win_length": win_length,
        "window": window,
        "center": center,
        "length": length,
    }
    ref_kwargs = {**kwargs, "window": utils.to_reference(window)}
    expected = torch.istft(utils.to_reference(inp), **ref_kwargs)

    with flag_gems.use_gems():
        actual = torch.istft(inp, **kwargs)

    utils.gems_assert_close(actual, expected, torch.float32, reduce_dim=n_fft)


@pytest.mark.istft
@pytest.mark.parametrize("batched", [False, True])
def test_istft_full_spectrum_complex_output(batched):
    # A 128-bin input exercises the non-onesided complex reconstruction path.
    shape = (2, 128, 10) if batched else (128, 10)
    inp = _complex_randn(shape)
    window = torch.ones(128, device=flag_gems.device)
    kwargs = {
        "n_fft": 128,
        "hop_length": 32,
        "window": window,
        "onesided": False,
        "return_complex": True,
    }
    expected = torch.istft(
        utils.to_reference(inp),
        **{**kwargs, "window": utils.to_reference(window)},
    )

    with flag_gems.use_gems():
        actual = torch.istft(inp, **kwargs)

    utils.gems_assert_close(actual, expected, torch.complex64, reduce_dim=128)


@pytest.mark.istft
def test_istft_default_rectangular_window():
    inp = _complex_randn((65, 8))
    expected = torch.istft(utils.to_reference(inp), n_fft=128)

    with flag_gems.use_gems():
        actual = torch.istft(inp, n_fft=128)

    utils.gems_assert_close(actual, expected, torch.float32, reduce_dim=128)


@pytest.mark.istft
def test_istft_checks_nola_condition():
    inp = _complex_randn((65, 8))
    window = torch.hann_window(128, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="window overlap add"):
        with flag_gems.use_gems():
            torch.istft(inp, n_fft=128, window=window, center=False)
