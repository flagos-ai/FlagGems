import pytest
import torch

import flag_gems
from flag_gems.ops.pixel_shuffle import pixel_shuffle
from flag_gems.testing import assert_close

FLOAT_DTYPES = [torch.float16, torch.float32]
UPSCALE_FACTORS = [2, 3, 4]

SHAPES = [
    (1, 4, 1, 1),
    (1, 4, 8, 8),
    (2, 4, 16, 16),
    (4, 9, 32, 32),
    (2, 16, 64, 64),
    (1, 4, 256, 256),
    (2, 9, 128, 128),
]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("r", UPSCALE_FACTORS)
def test_accuracy_pixel_shuffle(dtype, shape, r):
    N, C, H, W = shape
    C_in = C * r * r
    inp = torch.randn(N, C_in, H, W, dtype=dtype, device="cuda")
    ref = torch.pixel_shuffle(inp, r)
    with flag_gems.use_gems():
        out = pixel_shuffle(inp, r)
    assert_close(out, ref, dtype)


def test_accuracy_pixel_shuffle_large():
    inp = torch.randn(2, 36, 64, 64, dtype=torch.float32, device="cuda")
    ref = torch.pixel_shuffle(inp, 6)
    out = pixel_shuffle(inp, 6)
    assert_close(out, ref, torch.float32)


def test_accuracy_pixel_shuffle_zeros():
    inp = torch.zeros(2, 4, 8, 8, dtype=torch.float32, device="cuda")
    ref = torch.pixel_shuffle(inp, 2)
    out = pixel_shuffle(inp, 2)
    assert_close(out, ref, torch.float32)
