# tests/test_is_strides_like_format_random.py
import random
import pytest
import torch
import flag_gems

# ---------- random shape producer ----------
def generate_random_shape(min_dims=2, max_dims=4, min_dim_size=1, max_dim_size=8):
    ndims = random.randint(min_dims, max_dims)
    return tuple(random.randint(min_dim_size, max_dim_size) for _ in range(ndims))

# ---------- random test: contiguous ----------
def test_contiguous_random():
    num_examples = 100
    for _ in range(num_examples):
        shape = generate_random_shape()
        x = torch.randn(shape)  #  CPU
        with flag_gems.use_gems():
            result = flag_gems.is_strides_like_format(x, "contiguous")
        assert result.item() is True, f"Expected True for contiguous tensor with shape {shape}, got {result.item()}"

        # produce incontiguous tensor
        if x.dim() == 2:
            x_t = x.t()
        else:
            perm = list(range(x.dim()))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            x_t = x.permute(*perm)
        if not x_t.is_contiguous():
            with flag_gems.use_gems():
                result2 = flag_gems.is_strides_like_format(x_t, "contiguous")
            assert result2.item() is False, f"Expected False for non‑contiguous tensor with shape {shape}, got {result2.item()}"

# ---------- random test: any ----------
def test_any_random():
    num_examples = 100
    for _ in range(num_examples):
        shape = generate_random_shape()
        x = torch.randn(shape)
        with flag_gems.use_gems():
            result = flag_gems.is_strides_like_format(x, "any")
        assert result.item() is True, f"Expected True for any tensor with shape {shape}, got {result.item()}"

        if x.dim() == 2:
            x_t = x.t()
        else:
            perm = list(range(x.dim()))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            x_t = x.permute(*perm)
        with flag_gems.use_gems():
            result2 = flag_gems.is_strides_like_format(x_t, "any")
        assert result2.item() is True, f"Expected True for non‑contiguous any tensor with shape {shape}, got {result2.item()}"

# ---------- random test: channels_last ----------
def test_channels_last_random():
    """random test: channels_last case"""
    num_examples = 50
    for _ in range(num_examples):
        shape = generate_random_shape()
        # un_4D tensor return False
        if len(shape) != 4:
            x = torch.randn(shape)
            with flag_gems.use_gems():
                result = flag_gems.is_strides_like_format(x, "channels_last")
            assert result.item() is False, f"Expected False for non-4D shape {shape}, got {result.item()}"
            continue

        # 4D tensor：compare the result of operator with PyTorch's judge
        x = torch.randn(shape)
        # calculate PyTorch' result
        expected = x.is_contiguous(memory_format=torch.channels_last)
        with flag_gems.use_gems():
            actual = flag_gems.is_strides_like_format(x, "channels_last")
        assert actual.item() == expected, \
            f"Shape {shape}: expected {expected}, got {actual.item()}"

        x_cl = x.contiguous(memory_format=torch.channels_last)
        expected_cl = x_cl.is_contiguous(memory_format=torch.channels_last)
        with flag_gems.use_gems():
            actual_cl = flag_gems.is_strides_like_format(x_cl, "channels_last")
        assert actual_cl.item() == expected_cl, \
            f"Shape {shape} after conversion: expected {expected_cl}, got {actual_cl.item()}"

# ---------- abnormal test ----------
def test_unsupported_format():
    x = torch.randn(2, 3)
    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="Unsupported stride format"):
            flag_gems.is_strides_like_format(x, "invalid_format")
