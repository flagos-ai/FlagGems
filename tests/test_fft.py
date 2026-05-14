import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FFT_SHAPES = [
    (128, 64),
    (128, 128),
    (128, 256),
    (128, 512),
    (128, 1024),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 256),
]


@pytest.mark.fft
@pytest.mark.parametrize("shape", FFT_SHAPES)
def test_fft(shape):
    m, n = shape
    real = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)
    imag = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)
    x = torch.complex(real, imag)

    ref_x = utils.to_reference(x)
    ref_out = torch.fft.fft(ref_x)

    res_out = flag_gems.ops.fft(x)

    utils.gems_assert_close(res_out, ref_out, torch.complex64, reduce_dim=n)
