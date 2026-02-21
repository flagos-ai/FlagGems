"""Test cases for pixel_shuffle operator."""
import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, INT_DTYPES, gems_assert_equal, to_reference
from .conftest import QUICK_MODE

# Test configurations for pixel_shuffle
# Format: (input_shape, upscale_factor)
PIXEL_SHUFFLE_CONFIGS = [
    # ===== Small sizes =====
    ((1, 4, 2, 2), 2),  # Minimal: 1x4x2x2 -> 1x1x4x4
    ((1, 9, 4, 4), 3),  # Small: 1x9x4x4 -> 1x1x12x12
    ((2, 16, 8, 8), 2),  # Small batch: 2x16x8x8 -> 2x4x16x16
    # ===== Medium sizes =====
    ((1, 64, 16, 16), 2),  # Medium: 1x64x16x16 -> 1x16x32x32
    ((4, 36, 32, 32), 2),  # Medium batch: 4x36x32x32 -> 4x9x64x64
    ((2, 144, 64, 64), 3),  # Medium with r=3: 2x144x64x64 -> 2x16x192x192
    # ===== Large sizes =====
    ((1, 256, 128, 128), 2),  # Large: 1x256x128x128 -> 1x64x256x256
    ((1, 64, 256, 256), 2),  # Large spatial: 1x64x256x256 -> 1x16x512x512
    # ===== Different upscale factors =====
    ((1, 16, 16, 16), 4),  # r=4: 1x16x16x16 -> 1x1x64x64
    ((1, 25, 8, 8), 5),  # r=5: 1x25x8x8 -> 1x1x40x40
    # ===== Different channel counts =====
    ((1, 12, 16, 16), 2),  # C=3: 1x12x16x16 -> 1x3x32x32
    ((1, 108, 16, 16), 3),  # C=12: 1x108x16x16 -> 1x12x48x48
    # ===== Batch dimensions =====
    ((8, 36, 16, 16), 2),  # Larger batch: 8x36x16x16 -> 8x9x32x32
]

# Quick mode: use fewer test cases
if QUICK_MODE:
    PIXEL_SHUFFLE_CONFIGS = [
        PIXEL_SHUFFLE_CONFIGS[0],
        PIXEL_SHUFFLE_CONFIGS[4],
    ]
    FLOAT_DTYPES_TEST = [torch.float32]
else:
    FLOAT_DTYPES_TEST = FLOAT_DTYPES


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("shape, upscale_factor", PIXEL_SHUFFLE_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle(shape, upscale_factor, dtype):
    """Test pixel_shuffle accuracy."""
    _ = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(_, False)

    # PyTorch reference
    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, upscale_factor)

    # FlagGems implementation
    res_out = flag_gems.pixel_shuffle(_, upscale_factor)

    # For layout operations, we expect exact match
    gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle_edge_cases(dtype):
    """Test edge cases for pixel_shuffle."""
    # Test with ones (uniform values)
    _ = torch.ones((1, 4, 4, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(_, False)

    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
    res_out = flag_gems.pixel_shuffle(_, 2)
    gems_assert_equal(res_out, ref_out)

    # Test with zeros
    _ = torch.zeros((1, 9, 8, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(_, False)

    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 3)
    res_out = flag_gems.pixel_shuffle(_, 3)
    gems_assert_equal(res_out, ref_out)

    # Test with negative values
    _ = torch.randn((1, 16, 8, 8), dtype=dtype, device=flag_gems.device) - 5.0
    ref_inp = to_reference(_, False)

    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
    res_out = flag_gems.pixel_shuffle(_, 2)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle_different_upscale_factors(dtype):
    """Test pixel_shuffle with different upscale factors."""
    shape = (1, 64, 16, 16)
    _ = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    for r in [2, 4, 8]:
        # Adjust channels to be divisible by r^2
        channels = r * r * 4  # C = 4
        inp_adjusted = torch.randn(
            (1, channels, 16, 16), dtype=dtype, device=flag_gems.device
        )
        ref_inp = to_reference(inp_adjusted, False)

        ref_out = torch.nn.functional.pixel_shuffle(ref_inp, r)
        res_out = flag_gems.pixel_shuffle(inp_adjusted, r)

        gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle_different_batch_sizes(dtype):
    """Test pixel_shuffle with different batch sizes."""
    for batch_size in [1, 2, 4, 8, 16]:
        _ = torch.randn((batch_size, 36, 16, 16), dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(_, False)

        ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
        res_out = flag_gems.pixel_shuffle(_, 2)

        gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle_different_spatial_sizes(dtype):
    """Test pixel_shuffle with different spatial dimensions."""
    spatial_sizes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)]

    for h, w in spatial_sizes:
        _ = torch.randn((1, 16, h, w), dtype=dtype, device=flag_gems.device)
        ref_inp = to_reference(_, False)

        ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
        res_out = flag_gems.pixel_shuffle(_, 2)

        gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
def test_accuracy_pixel_shuffle_large_input(dtype):
    """Test pixel_shuffle with large input sizes."""
    # Large spatial dimensions
    _ = torch.randn((1, 64, 512, 512), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(_, False)

    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
    res_out = flag_gems.pixel_shuffle(_, 2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.pixel_shuffle
def test_pixel_shuffle_error_handling():
    """Test error handling for invalid inputs."""
    # Test with invalid upscale_factor
    _ = torch.randn((1, 16, 8, 8), device=flag_gems.device)

    with pytest.raises(ValueError, match="upscale_factor must be positive"):
        flag_gems.pixel_shuffle(_, 0)

    with pytest.raises(ValueError, match="upscale_factor must be positive"):
        flag_gems.pixel_shuffle(_, -1)

    # Test with channels not divisible by r^2
    _ = torch.randn((1, 15, 8, 8), device=flag_gems.device)

    with pytest.raises(ValueError, match="divisible by"):
        flag_gems.pixel_shuffle(_, 2)  # 15 is not divisible by 4

    # Test with insufficient dimensions
    _ = torch.randn((16, 8), device=flag_gems.device)

    with pytest.raises(ValueError, match="at least 3 dimensions"):
        flag_gems.pixel_shuffle(_, 2)


@pytest.mark.pixel_shuffle
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_pixel_shuffle_integer_types(dtype):
    """Test pixel_shuffle with integer data types."""
    _ = torch.randint(0, 100, (1, 16, 8, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(_, False)

    ref_out = torch.nn.functional.pixel_shuffle(ref_inp, 2)
    res_out = flag_gems.pixel_shuffle(_, 2)

    gems_assert_equal(res_out, ref_out)
