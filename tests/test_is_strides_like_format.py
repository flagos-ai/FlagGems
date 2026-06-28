# tests/test_is_strides_like_format_random.py
import random
import pytest
import torch
import flag_gems

# ---------- 随机形状生成器 ----------
def generate_random_shape(min_dims=2, max_dims=4, min_dim_size=1, max_dim_size=8):
    ndims = random.randint(min_dims, max_dims)
    return tuple(random.randint(min_dim_size, max_dim_size) for _ in range(ndims))

# ---------- 随机测试 contiguous ----------
def test_contiguous_random():
    num_examples = 100
    for _ in range(num_examples):
        shape = generate_random_shape()
        x = torch.randn(shape)  # 默认 CPU
        with flag_gems.use_gems():
            result = flag_gems.is_strides_like_format(x, "contiguous")
        assert result.item() is True, f"Expected True for contiguous tensor with shape {shape}, got {result.item()}"

        # 制造非连续张量
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

# ---------- 随机测试 any ----------
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

# ---------- 随机测试 channels_last（移除 skipif，因为 CPU 总是支持）----------
def test_channels_last_random():
    """随机测试 channels_last 分支（固定运行在 CPU 上）"""
    num_examples = 50
    for _ in range(num_examples):
        shape = generate_random_shape()
        # 非 4D 张量必须返回 False
        if len(shape) != 4:
            x = torch.randn(shape)
            with flag_gems.use_gems():
                result = flag_gems.is_strides_like_format(x, "channels_last")
            assert result.item() is False, f"Expected False for non-4D shape {shape}, got {result.item()}"
            continue

        # 4D 张量：比较算子结果与 PyTorch 的判断
        x = torch.randn(shape)
        # 计算 PyTorch 的标准结果
        expected = x.is_contiguous(memory_format=torch.channels_last)
        with flag_gems.use_gems():
            actual = flag_gems.is_strides_like_format(x, "channels_last")
        assert actual.item() == expected, \
            f"Shape {shape}: expected {expected}, got {actual.item()}"

        # 测试转换为 channels_last 后的张量
        # 注意：在 CPU 上转换总是成功（但某些形状可能转换后仍与 contiguous 相同）
        x_cl = x.contiguous(memory_format=torch.channels_last)
        expected_cl = x_cl.is_contiguous(memory_format=torch.channels_last)
        with flag_gems.use_gems():
            actual_cl = flag_gems.is_strides_like_format(x_cl, "channels_last")
        assert actual_cl.item() == expected_cl, \
            f"Shape {shape} after conversion: expected {expected_cl}, got {actual_cl.item()}"

# ---------- 异常测试 ----------
def test_unsupported_format():
    x = torch.randn(2, 3)
    with flag_gems.use_gems():
        with pytest.raises(ValueError, match="Unsupported stride format"):
            flag_gems.is_strides_like_format(x, "invalid_format")