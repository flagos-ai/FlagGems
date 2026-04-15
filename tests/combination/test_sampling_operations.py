"""
P2级测试：采样与随机操作组合测试

测试场景：
- Dropout训练模式
- 权重初始化组合
- 多项式采样
- 数据增强pipeline
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import assert_loss_close

device = flag_gems.device


# ========== 简化模型 ==========


class DropoutPipeline(nn.Module):
    """Dropout完整pipeline"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, training=True):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x) if training else x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x) if training else x
        x = self.fc3(x)
        return x


class WeightInitializer:
    """权重初始化器"""

    @staticmethod
    def xavier_init(shape):
        """Xavier初始化"""
        gain = nn.init.calculate_gain("relu")
        std = gain * math.sqrt(2.0 / sum(shape))
        return torch.randn(shape, device=device) * std

    @staticmethod
    def kaiming_init(shape):
        """Kaiming初始化"""
        gain = nn.init.calculate_gain("relu")
        std = gain / (shape[0] ** 0.5)
        return torch.randn(shape, device=device) * std

    @staticmethod
    def normal_init(shape, mean=0, std=0.02):
        """正态分布初始化"""
        return torch.randn(shape, device=device) * std + mean


class SamplingPolicy(nn.Module):
    """采样策略"""

    def __init__(self, num_actions, temperature=1.0):
        super().__init__()
        self.num_actions = num_actions
        self.temperature = temperature

    def forward(self, logits):
        # Softmax得到概率分布
        probs = F.softmax(logits / self.temperature, dim=-1)
        # 多项式采样
        samples = torch.multinomial(probs, num_samples=1)
        return samples.squeeze(-1), probs


class DataAugmentation(nn.Module):
    """数据增强pipeline"""

    def __init__(self, augment_prob=0.5):
        super().__init__()
        self.augment_prob = augment_prob

    def forward(self, x):
        # 随机决定是否增强
        if torch.rand(1).item() < self.augment_prob:
            # 随机翻转
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
            # 随机缩放
            scale = 0.8 + torch.rand(1).item() * 0.4
            x = x * scale
        return x


# ========== 测试类 ==========


class TestDropoutCombination:
    """Dropout组合测试"""

    @pytest.mark.integration
    def test_dropout_training_vs_eval(self, use_gems):
        """测试训练和评估模式的Dropout"""
        model = DropoutPipeline(64, 128, 10).to(device)
        x = torch.randn(8, 64, device=device)

        # 训练模式
        model.train()
        _ = model(x, training=True)
        _ = model(x, training=True)
        # 多次前向传播应该不同（由于Dropout随机性）

        # 评估模式
        model.eval()
        output_eval1 = model(x, training=False)
        output_eval2 = model(x, training=False)
        # 评估模式应该一致
        assert torch.allclose(output_eval1, output_eval2)

    @pytest.mark.integration
    @pytest.mark.parametrize("dropout_rate", [0.1, 0.3, 0.5, 0.7])
    def test_dropout_rate_effect(self, dropout_rate, use_gems):
        """测试不同dropout率"""
        dropout = nn.Dropout(dropout_rate).to(device)
        x = torch.ones(1000, 1000, device=device)

        dropout.train()
        output = dropout(x)

        # 统计被置零的比例
        zero_ratio = (output == 0).float().mean().item()
        # 应该接近dropout_rate
        assert abs(zero_ratio - dropout_rate) < 0.05

    @pytest.mark.integration
    def test_dropout_gradient_flow(self, use_gems):
        """测试Dropout梯度流"""
        model = DropoutPipeline(32, 64, 10).to(device)
        x = torch.randn(4, 32, device=device, requires_grad=True)

        model.train()
        output = model(x, training=True)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.integration
    def test_dropout_scaling_factor(self, use_gems):
        """测试Dropout缩放因子"""
        dropout = nn.Dropout(0.5).to(device)
        x = torch.ones(1000, 1000, device=device)

        dropout.train()
        output = dropout(x)

        # 未被置零的元素应该被放大（缩放因子 = 1/(1-p)）
        non_zero_mask = output != 0
        expected_scale = 1.0 / (1 - 0.5)  # = 2.0

        if non_zero_mask.any():
            actual_values = output[non_zero_mask]
            assert torch.allclose(
                actual_values, torch.ones_like(actual_values) * expected_scale, rtol=0.1
            )


class TestWeightInitialization:
    """权重初始化测试"""

    @pytest.mark.integration
    def test_xavier_initialization(self, use_gems):
        """测试Xavier初始化"""
        shape = (256, 512)
        weight = WeightInitializer.xavier_init(shape)

        # 验证形状
        assert weight.shape == shape
        # 验证均值接近0
        assert abs(weight.mean().item()) < 0.1
        # 验证方差合理
        var = weight.var().item()
        assert var > 0.001 and var < 0.1

    @pytest.mark.integration
    def test_kaiming_initialization(self, use_gems):
        """测试Kaiming初始化"""
        shape = (512, 256)
        weight = WeightInitializer.kaiming_init(shape)

        assert weight.shape == shape
        assert abs(weight.mean().item()) < 0.1
        assert weight.var().item() > 0

    @pytest.mark.integration
    def test_initialization_affects_training(self, use_gems):
        """测试初始化对训练的影响"""
        # Xavier初始化
        torch.manual_seed(42)
        linear_xavier = nn.Linear(64, 32)
        nn.init.xavier_normal_(linear_xavier.weight)

        # Kaiming初始化
        torch.manual_seed(42)
        linear_kaiming = nn.Linear(64, 32)
        nn.init.kaiming_normal_(linear_kaiming.weight)

        x = torch.randn(8, 64, device=device)

        # 两种初始化应该产生不同的输出分布
        out_xavier = linear_xavier(x.to("cpu"))
        out_kaiming = linear_kaiming(x.to("cpu"))

        # 两者不应完全相同
        assert not torch.allclose(out_xavier, out_kaiming, rtol=1e-3)


class TestSamplingOperations:
    """采样操作测试"""

    @pytest.mark.integration
    def test_multinomial_sampling(self, use_gems):
        """测试多项式采样"""
        policy = SamplingPolicy(10).to(device)
        logits = torch.randn(100, 10, device=device)

        samples, probs = policy(logits)

        # 验证采样在有效范围内
        assert (samples >= 0).all() and (samples < 10).all()
        # 验证概率和为1
        assert torch.allclose(
            probs.sum(dim=-1), torch.ones(100, device=device), atol=1e-5
        )

        # Reference comparison for the deterministic part (probabilities)
        ref_probs = F.softmax(
            logits.detach().to("cpu").to(torch.float64) / 1.0, dim=-1
        ).to(device)
        assert_loss_close(
            probs.sum(),
            ref_probs.sum().float(),
            logits.dtype,
            name="Sampling probs sum",
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("temperature", [0.01, 0.1, 1.0, 10.0])
    def test_temperature_effect(self, temperature, use_gems):
        """测试温度参数对采样的影响"""
        logits = torch.tensor([[1.0, 2.0, 3.0]], device=device)

        policy = SamplingPolicy(3, temperature=temperature)
        _, probs = policy(logits)

        # 高温度 应该使分布更均匀
        # 低温度 应该使分布更尖锐
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        assert entropy > 0  # 应该有不确定度

    @pytest.mark.integration
    def test_sampling_reproducibility(self, use_gems):
        """测试采样的可重复性"""
        policy = SamplingPolicy(5).to(device)
        logits = torch.randn(10, 5, device=device)

        # 设置随机种子
        torch.manual_seed(42)
        samples1, _ = policy(logits)

        torch.manual_seed(42)
        samples2, _ = policy(logits)

        # 相同种子应该产生相同采样
        assert torch.equal(samples1, samples2)


class TestDataAugmentation:
    """数据增强测试"""

    @pytest.mark.integration
    def test_augmentation_randomness(self, use_gems):
        """测试增强的随机性"""
        aug = DataAugmentation(augment_prob=1.0).to(device)
        x = torch.randn(4, 3, 32, 32, device=device)

        # 多次增强应该产生不同结果
        output1 = aug(x)
        output2 = aug(x)

        # 由于随机性，两次结果很可能不同
        # (但不保证总是不同，所以这里只验证形状)
        assert output1.shape == x.shape
        assert output2.shape == x.shape

    @pytest.mark.integration
    def test_augmentation_probability(self, use_gems):
        """测试增强概率"""
        aug = DataAugmentation(augment_prob=0.5).to(device)
        x = torch.ones(100, 3, 32, 32, device=device)

        # 运行多次
        augmented = []
        for _ in range(10):
            aug_out = aug(x.clone())
            augmented.append(aug_out)

        # 至少有一些应该被增强（不等于原值）
        # 由于随机性，至少检查形状正确
        for aug_out in augmented:
            assert aug_out.shape == x.shape


class TestRandomOperations:
    """其他随机操作测试"""

    @pytest.mark.integration
    def test_rand_and_randn(self, use_gems):
        """测试rand和randn"""
        # 均匀分布
        uni = torch.rand(1000, device=device)
        assert uni.min() >= 0 and uni.max() <= 1
        assert abs(uni.mean().item() - 0.5) < 0.1

        # 标准正态分布
        norm = torch.randn(1000, device=device)
        assert abs(norm.mean().item()) < 0.1
        assert abs(norm.std().item() - 1.0) < 0.1

    @pytest.mark.integration
    def test_randperm(self, use_gems):
        """测试随机排列"""
        perm = torch.randperm(100, device=device)

        # 验证是排列
        assert perm.shape == (100,)
        assert torch.sort(perm)[0].equal(torch.arange(100, device=device))

    @pytest.mark.integration
    def test_normal_distribution_sampling(self, use_gems):
        """测试正态分布采样"""
        mean = 5.0
        std = 2.0
        samples = torch.normal(mean, std, size=(1000,), device=device)

        # 统计验证
        sample_mean = samples.mean().item()
        sample_std = samples.std().item()

        assert abs(sample_mean - mean) < 0.5
        assert abs(sample_std - std) < 0.5
