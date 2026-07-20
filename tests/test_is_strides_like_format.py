# tests/test_is_strides_like_format.py
import random

import pytest
import torch

import flag_gems

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

_FORMAT_MAP = {
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}


def _torch_is_strides_like_format(x, fmt):
    if fmt not in _FORMAT_MAP:
        return False
    return torch.ops.aten.is_strides_like_format(x, _FORMAT_MAP[fmt])


# ---------- Fixed-shape parameterized tests (preserved) ----------
SHAPES = [
    (2, 3),
    (4, 5, 6),
    (2, 3, 4, 5),
    (8, 3, 224, 224),
    (2, 3, 4, 5, 6),
]
FORMATS = ["channels_last", "channels_last_3d"]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fmt", FORMATS)
def test_is_strides_like_format(shape, fmt):
    x = torch.randn(shape)
    expected = _torch_is_strides_like_format(x, fmt)
    with flag_gems.use_gems():
        actual = flag_gems.is_strides_like_format(x, fmt)
    assert actual == expected, (
        f"Shape {shape}, strides {x.stride()}, format {fmt}: "
        f"expected {expected}, got {actual}"
    )


# ---------- Random shape generator ----------
def generate_random_shape(max_elements=2_000_000):
    """Generate random shapes with 2~5 dims, each dim size 1~128 (occasionally larger)."""
    ndims = random.randint(2, 5)
    while True:
        dim_sizes = []
        for _ in range(ndims):
            if random.random() < 0.2:  # 20% chance for a larger dimension
                dim_sizes.append(random.choice([64, 128, 256]))
            else:
                dim_sizes.append(random.randint(1, 64))
        shape = tuple(dim_sizes)
        total = 1
        for d in shape:
            total *= d
        if total <= max_elements:
            return shape


# ---------- Random test ----------
@pytest.mark.is_strides_like_format_random
def test_is_strides_like_format_random():
    """Randomized correctness test with 100 random shapes, including after format conversion."""
    num_examples = 100
    for _ in range(num_examples):
        shape = generate_random_shape()
        x = torch.randn(shape)
        ndim = len(shape)

        # Test channels_last (only valid for 4D)
        fmt = "channels_last"
        expected = _torch_is_strides_like_format(x, fmt)
        with flag_gems.use_gems():
            actual = flag_gems.is_strides_like_format(x, fmt)
        assert actual == expected, (
            f"Shape {shape}, strides {x.stride()}, format {fmt}: "
            f"expected {expected}, got {actual}"
        )

        # Test channels_last_3d (only valid for 5D)
        fmt = "channels_last_3d"
        expected = _torch_is_strides_like_format(x, fmt)
        with flag_gems.use_gems():
            actual = flag_gems.is_strides_like_format(x, fmt)
        assert actual == expected, (
            f"Shape {shape}, strides {x.stride()}, format {fmt}: "
            f"expected {expected}, got {actual}"
        )

        # Test after explicit conversion (if applicable)
        if ndim == 4:
            x_cl = x.contiguous(memory_format=torch.channels_last)
            expected = _torch_is_strides_like_format(x_cl, "channels_last")
            with flag_gems.use_gems():
                actual = flag_gems.is_strides_like_format(x_cl, "channels_last")
            assert actual == expected, (
                f"Shape {shape}, strides {x_cl.stride()} (after channels_last conversion): "
                f"expected {expected}, got {actual}"
            )
            # Converted tensor should return False for channels_last_3d
            expected_3d = _torch_is_strides_like_format(x_cl, "channels_last_3d")
            with flag_gems.use_gems():
                actual_3d = flag_gems.is_strides_like_format(x_cl, "channels_last_3d")
            assert actual_3d == expected_3d, (
                f"Shape {shape} after channels_last conversion, format channels_last_3d: "
                f"expected {expected_3d}, got {actual_3d}"
            )

        elif ndim == 5:
            x_cl = x.contiguous(memory_format=torch.channels_last_3d)
            expected = _torch_is_strides_like_format(x_cl, "channels_last_3d")
            with flag_gems.use_gems():
                actual = flag_gems.is_strides_like_format(x_cl, "channels_last_3d")
            assert actual == expected, (
                f"Shape {shape}, strides {x_cl.stride()} (after channels_last_3d conversion): "
                f"expected {expected}, got {actual}"
            )
            # Converted tensor should return False for channels_last
            expected_cl = _torch_is_strides_like_format(x_cl, "channels_last")
            with flag_gems.use_gems():
                actual_cl = flag_gems.is_strides_like_format(x_cl, "channels_last")
            assert actual_cl == expected_cl, (
                f"Shape {shape} after channels_last_3d conversion, format channels_last: "
                f"expected {expected_cl}, got {actual_cl}"
            )

        # Test unsupported formats should return False
        for fmt in ["contiguous", "any", "invalid_format"]:
            with flag_gems.use_gems():
                actual = flag_gems.is_strides_like_format(x, fmt)
            assert (
                actual is False
            ), f"Shape {shape}, format {fmt}: expected False, got {actual}"


# ---------- Test unsupported formats (standalone) ----------
def test_unsupported_format():
    x = torch.randn(2, 3)
    with flag_gems.use_gems():
        for fmt in ["contiguous", "any", "invalid_format"]:
            actual = flag_gems.is_strides_like_format(x, fmt)
            assert actual is False, f"Expected False for format {fmt}, got {actual}"
