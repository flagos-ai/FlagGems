import pytest
import torch

import flag_gems
from flag_gems.ops.pixel_unshuffle import pixel_unshuffle

FLOAT_DTYPES = [torch.float16, torch.float32]
UPSCALE_FACTORS = [2, 3, 4]
SHAPES = [
    (1, 1, 2, 2),
    (1, 1, 8, 8),
    (2, 1, 16, 16),
    (1, 3, 32, 32),
    (2, 3, 64, 64),
    (4, 8, 128, 128),
    (2, 16, 256, 256),
]


@pytest.mark.parametrize("r", UPSCALE_FACTORS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pixel_unshuffle(dtype, shape, r):
    N, C, H, W = shape
    H_in = H * r
    W_in = W * r
    inp = torch.randn(N, C, H_in, W_in, dtype=dtype, device="cuda")
    ref = torch.pixel_unshuffle(inp, r)
    with flag_gems.use_gems():
        out = pixel_unshuffle(inp, r)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_accuracy_pixel_unshuffle_large():
    inp = torch.randn(2, 3, 512, 512, dtype=torch.float32, device="cuda")
    ref = torch.pixel_unshuffle(inp, 4)
    with flag_gems.use_gems():
        out = pixel_unshuffle(inp, 4)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_accuracy_pixel_unshuffle_zeros():
    inp = torch.zeros(1, 1, 4, 4, dtype=torch.float32, device="cuda")
    ref = torch.pixel_unshuffle(inp, 2)
    with flag_gems.use_gems():
        out = pixel_unshuffle(inp, 2)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
