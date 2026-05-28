import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.istft
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", utils.ISTFT_SHAPES)
def test_istft_basic(dtype, shape):
    """Test basic istft functionality."""
    n_fft = shape["n_fft"]
    hop_length = shape["hop_length"]
    n_frames = shape["n_frames"]

    batch_size = 1
    n_freq = n_fft // 2 + 1  # onesided

    inp = torch.randn(
        batch_size, n_freq, n_frames, dtype=dtype, device=flag_gems.device
    )
    inp_complex = torch.complex(inp, torch.randn_like(inp))

    window = torch.hann_window(n_fft, dtype=dtype, device=flag_gems.device)

    ref_out = torch.istft(
        inp_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    with flag_gems.use_gems():
        gems_out = torch.istft(
            inp_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )

    torch.testing.assert_close(gems_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.istft
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shape", utils.ISTFT_SHAPES)
def test_istft_normalized(dtype, shape):
    """Test istft with normalized=True."""
    n_fft = shape["n_fft"]
    hop_length = shape["hop_length"]
    n_frames = shape["n_frames"]

    batch_size = 1
    n_freq = n_fft // 2 + 1

    inp = torch.randn(
        batch_size, n_freq, n_frames, dtype=dtype, device=flag_gems.device
    )
    inp_complex = torch.complex(inp, torch.randn_like(inp))

    window = torch.hann_window(n_fft, dtype=dtype, device=flag_gems.device)

    ref_out = torch.istft(
        inp_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        normalized=True,
        onesided=True,
        return_complex=False,
    )

    with flag_gems.use_gems():
        gems_out = torch.istft(
            inp_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            normalized=True,
            onesided=True,
            return_complex=False,
        )

    torch.testing.assert_close(gems_out, ref_out, rtol=1e-2, atol=1e-2)
