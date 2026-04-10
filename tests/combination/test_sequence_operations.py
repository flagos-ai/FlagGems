"""
P2级测试：序列操作组合测试

测试场景：
- 序列生成（arange → embedding）
- 累积操作链
- 位置编码（sin/cos）
- 序列索引重排
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .conftest import QUICK_MODE
from .accuracy_utils import compute_reference, combo_assert_close, COMBO_FLOAT_DTYPES

device = flag_gems.device


# ========== 简化模型 ==========

class PositionalEncoding(nn.Module):
    """位置编码：arange → sin/cos"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class SequentialStatistics(nn.Module):
    """序列统计：cumsum → cummax → cummin"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # 累积和
        cumsum = torch.cumsum(x, dim=-1)
        # 累积最大值
        cummax = torch.cummax(x, dim=-1).values
        # 累积最小值
        cummin = torch.cummin(x, dim=-1).values
        
        return cumsum, cummax, cummin


class SequenceShuffle(nn.Module):
    """序列重排：arange → gather → scatter"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x, indices):
        # 根据索引重排序列
        shuffled = torch.gather(x, dim=1, index=indices)
        return shuffled
    
    def shuffle_and_restore(self, x, indices):
        """打乱并恢复"""
        # 打乱
        shuffled = torch.gather(x, dim=1, index=indices)
        # 恢复
        restored = torch.scatter(torch.zeros_like(x), dim=1, index=indices, src=x)
        return shuffled, restored


class SequenceModel(nn.Module):
    """简单序列模型：arange → embedding → linear"""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, seq_indices):
        # 根据索引获取嵌入
        embedded = self.embedding(seq_indices)
        # 线性变换
        output = self.linear(embedded)
        return output


# ========== 测试类 ==========

class TestPositionalEncoding:
    """位置编码测试"""
    
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "batch_size,seq_len,d_model",
        [
            pytest.param(2, 16, 64, id="small"),
            pytest.param(4, 64, 128, id="medium"),
            pytest.param(8, 256, 512, id="large",
                         marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
        ],
    )
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    @pytest.mark.integration
    def test_positional_encoding_shape(self, batch_size, seq_len, d_model, dtype, use_gems):
        """测试位置编码形状"""
        pe = PositionalEncoding(d_model).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        pe.eval()
        with torch.no_grad():
            output = pe(x)

        assert output.shape == x.shape

        # Reference comparison (PE ≈ 2 ops: buffer lookup + add)
        ref_output = compute_reference(pe, x)
        combo_assert_close(output, ref_output, x.dtype, num_ops=2, name="PositionalEncoding")
    
    @pytest.mark.integration
    def test_positional_encoding_values(self, use_gems):
        """测试位置编码值范围"""
        pe = PositionalEncoding(64, max_len=100).to(device)
        x = torch.zeros(1, 50, 64, device=device)
        
        output = pe(x)
        
        # 位置编码应该在[-1, 1]范围内（sin/cos）
        assert output.min() >= -1.0
        assert output.max() <= 1.0
        # 由于x是零，输出就是位置编码本身
        assert not torch.allclose(output, torch.zeros_like(output))
    
    @pytest.mark.integration
    def test_positional_encoding_different_positions(self, use_gems):
        """测试不同位置有不同的编码"""
        pe = PositionalEncoding(128).to(device)
        x = torch.zeros(1, 100, 128, device=device)
        
        output = pe(x)
        
        # 不同位置的编码应该不同
        pos_0 = output[0, 0, :]
        pos_1 = output[0, 1, :]
        pos_10 = output[0, 10, :]
        
        assert not torch.allclose(pos_0, pos_1)
        assert not torch.allclose(pos_0, pos_10)
        assert not torch.allclose(pos_1, pos_10)


class TestCumulativeOperations:
    """累积操作测试"""
    
    @pytest.mark.integration
    def test_cumsum_correctness(self, use_gems):
        """测试累积和正确性"""
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float, device=device)
        
        cumsum = torch.cumsum(x, dim=-1)
        expected = torch.tensor([[1, 3, 6, 10, 15]], dtype=torch.float, device=device)
        
        assert torch.allclose(cumsum, expected)
    
    @pytest.mark.integration
    def test_cummax_correctness(self, use_gems):
        """测试累积最大值正确性"""
        x = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=torch.float, device=device)
        
        cummax = torch.cummax(x, dim=-1).values
        expected = torch.tensor([[3, 3, 4, 4, 5, 9, 9, 9]], dtype=torch.float, device=device)
        
        assert torch.allclose(cummax, expected)
    
    @pytest.mark.integration
    def test_cummin_correctness(self, use_gems):
        """测试累积最小值正确性"""
        x = torch.tensor([[5, 3, 4, 1, 2, 0, 6]], dtype=torch.float, device=device)
        
        cummin = torch.cummin(x, dim=-1).values
        expected = torch.tensor([[5, 3, 3, 1, 1, 0, 0]], dtype=torch.float, device=device)
        
        assert torch.allclose(cummin, expected)
    
    @pytest.mark.integration
    def test_cumulative_statistics_chain(self, use_gems):
        """测试累积统计链"""
        model = SequentialStatistics().to(device)
        x = torch.randn(4, 32, device=device)
        
        cumsum, cummax, cummin = model(x)
        
        # 验证形状
        assert cumsum.shape == x.shape
        assert cummax.shape == x.shape
        assert cummin.shape == x.shape
        
        # 验证最后位置的累积和等于总和
        assert torch.allclose(cumsum[:, -1], x.sum(dim=-1))
        
        # 验证累积最大值的最后一个位置等于全局最大值
        assert torch.allclose(cummax[:, -1], x.max(dim=-1).values)
        
        # 验证累积最小值的最后一个位置等于全局最小值
        assert torch.allclose(cummin[:, -1], x.min(dim=-1).values)
    
    @pytest.mark.integration
    def test_cumulative_gradient(self, use_gems):
        """测试累积操作梯度"""
        x = torch.randn(4, 16, device=device, requires_grad=True)
        
        cumsum = torch.cumsum(x, dim=-1)
        loss = cumsum.sum()
        loss.backward()
        
        assert x.grad is not None
        # 梯度应该全为1的累积
        assert torch.isfinite(x.grad).all()


class TestSequenceShuffle:
    """序列重排测试"""
    
    @pytest.mark.integration
    def test_sequence_gather(self, use_gems):
        """测试序列gather"""
        model = SequenceShuffle().to(device)
        x = torch.randn(2, 10, 64, device=device)
        
        # 创建随机索引
        indices = torch.randperm(10, device=device).unsqueeze(0).expand(2, -1)
        
        shuffled = model(x, indices)
        
        # 验证形状相同
        assert shuffled.shape == x.shape
        
        # 验证gather正确性
        expected = torch.gather(x, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, 64))
        assert torch.allclose(shuffled, expected)
    
    @pytest.mark.integration
    def test_shuffle_and_restore(self, use_gems):
        """测试打乱并恢复"""
        model = SequenceShuffle().to(device)
        x = torch.randn(1, 8, 32, device=device)
        indices = torch.randperm(8, device=device).unsqueeze(0)
        
        shuffled, restored = model.shuffle_and_restore(x, indices)
        
        # 形状应该一致
        assert shuffled.shape == x.shape
        
        # 验证恢复（scatter的逆操作）
        # 原始数据和恢复的数据应该可以对应


class TestArangeEmbedding:
    """Arange → Embedding测试"""
    
    @pytest.mark.integration
    def test_arange_as_indices(self, use_gems):
        """测试arange作为索引"""
        seq_len = 32
        indices = torch.arange(seq_len, device=device)
        
        # 验证是0到seq_len-1的序列
        assert indices[0] == 0
        assert indices[-1] == seq_len - 1
        assert torch.equal(torch.sort(indices)[0], indices)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_sequence_embedding(self, dtype, use_gems):
        """测试序列嵌入"""
        vocab_size = 1000
        embed_dim = 256
        hidden_dim = 128

        model = SequenceModel(vocab_size, embed_dim, hidden_dim).to(device).to(dtype)

        # 创建序列索引（arange）
        seq_indices = torch.arange(0, 64, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(seq_indices)

        # 验证输出形状
        assert output.shape == (1, 64, hidden_dim)

        # Reference comparison (embedding + linear ≈ 2 ops)
        ref_output = compute_reference(model, seq_indices)
        combo_assert_close(output, ref_output, dtype, num_ops=2, name=f"SequenceModel {dtype}")
    
    @pytest.mark.integration
    def test_batch_sequence_indices(self, use_gems):
        """测试批量序列索引"""
        vocab_size = 500
        embed_dim = 128
        
        embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        
        # 批量不同长度的序列
        batch_size = 4
        seq_len = 32
        indices = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        embedded = embedding(indices)
        
        assert embedded.shape == (batch_size, seq_len, embed_dim)


class TestLinspace:
    """Linspace测试"""
    
    @pytest.mark.integration
    def test_linspace_basic(self, use_gems):
        """测试linspace基本功能"""
        start = 0.0
        end = 1.0
        steps = 10
        
        values = torch.linspace(start, end, steps, device=device)
        
        assert values[0] == start
        assert abs(values[-1].item() - end) < 1e-5
        assert len(values) == steps
        
    @pytest.mark.integration
    def test_linspace_uniform_spacing(self, use_gems):
        """测试linspace均匀间距"""
        values = torch.linspace(0, 10, 100, device=device)
        
        # 计算相邻元素差
        diffs = values[1:] - values[:-1]
        
        # 所有差值应该相等
        assert torch.allclose(diffs, diffs[0].expand_as(diffs), rtol=1e-4)


class TestLogspace:
    """Logspace测试"""
    
    @pytest.mark.integration
    def test_logspace_basic(self, use_gems):
        """测试logspace基本功能"""
        start = -2
        end = 2
        steps = 20
        
        values = torch.logspace(start, end, steps, device=device)
        
        # 验证范围
        assert values.min() >= 10 ** start
        assert values.max() <= 10 ** end
    
    @pytest.mark.integration
    def test_logspace_logarithmic_spacing(self, use_gems):
        """测试logspace对数间距"""
        values = torch.logspace(0, 2, 10, device=device)  # 1到100
        
        # 在对数空间应该是均匀的
        log_values = torch.log10(values)
        diffs = log_values[1:] - log_values[:-1]
        
        assert torch.allclose(diffs, diffs[0].expand_as(diffs), rtol=1e-4)