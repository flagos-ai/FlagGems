"""
P2级测试：损失函数组合测试

测试场景：
- CrossEntropy完整实现（softmax → log → nll_loss）
- MSE损失组合
- 多任务损失组合
- 损失函数梯度验证
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import assert_loss_close
from .conftest import QUICK_MODE

device = flag_gems.device


# ========== 简化模型 ==========


class CrossEntropyImpl(nn.Module):
    """CrossEntropy分步实现：softmax → log → nll_loss"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        # Step 1: softmax
        probs = F.softmax(logits, dim=-1)
        # Step 2: log
        log_probs = torch.log(probs + 1e-10)  # 数值稳定
        # Step 3: nll_loss
        loss = F.nll_loss(log_probs, target, reduction=self.reduction)
        return loss


class MultiTaskLoss(nn.Module):
    """多任务损失组合"""

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = torch.tensor(weights, device=device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

    def forward(
        self, logits_cls, target_cls, pred_reg, target_reg, anchor, positive, label
    ):
        loss1 = self.ce_loss(logits_cls, target_cls)
        loss2 = self.mse_loss(pred_reg, target_reg)
        loss3 = self.margin_loss(anchor, positive, label)

        # 加权组合
        total_loss = (
            self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3
        )
        return total_loss


class ContrastiveLoss(nn.Module):
    """对比学习损失"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 特征归一化
        features = F.normalize(features, dim=1)
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t())
        # 除以温度参数
        similarity = similarity / self.temperature
        # 构建正样本mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()
        # 计算损失（简化版）
        exp_sim = torch.exp(similarity)
        pos = exp_sim * mask
        neg = exp_sim * (1 - mask)
        loss = -torch.log(pos.sum(dim=1) / (pos.sum(dim=1) + neg.sum(dim=1) + 1e-10))
        return loss.mean()


# ========== 测试类 ==========


class TestCrossEntropyCombination:
    """CrossEntropy组合测试"""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "batch_size,num_classes,seq_len",
        [
            pytest.param(4, 10, 16, id="small"),
            pytest.param(16, 100, 32, id="medium"),
            pytest.param(
                32,
                1000,
                64,
                id="large",
                marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode"),
            ),
        ],
    )
    def test_cross_entropy_step_by_step(
        self, batch_size, num_classes, seq_len, use_gems
    ):
        """测试分步实现的CrossEntropy"""
        logits = torch.randn(batch_size, seq_len, num_classes, device=device)
        target = torch.randint(0, num_classes, (batch_size, seq_len), device=device)

        # 分步实现
        ce_impl = CrossEntropyImpl()
        loss_step = ce_impl(logits.view(-1, num_classes), target.view(-1))

        # PyTorch标准实现
        ce_standard = nn.CrossEntropyLoss()
        loss_standard = ce_standard(logits.view(-1, num_classes), target.view(-1))

        # 验证结果相近（使用 assert_loss_close 替代手写 isclose）
        assert_loss_close(
            loss_step, loss_standard, logits.dtype, name="CrossEntropy step-by-step"
        )

    @pytest.mark.integration
    @pytest.mark.numerical_stability
    def test_cross_entropy_large_logits(self, use_gems):
        """测试大logit值的数值稳定性"""
        logits = torch.randn(4, 10, device=device) * 100
        target = torch.randint(0, 10, (4,), device=device)

        ce = nn.CrossEntropyLoss()
        loss = ce(logits, target)

        # 应该产生有限的损失值
        assert torch.isfinite(loss)
        assert loss >= 0  # CrossEntropy总是非负

        # Reference comparison on CPU/fp64
        ref_logits = logits.detach().to("cpu").to(torch.float64)
        ref_target = target.to("cpu")
        ref_loss = nn.CrossEntropyLoss()(ref_logits, ref_target).to(device)
        assert_loss_close(
            loss, ref_loss, logits.dtype, name="CrossEntropy large logits"
        )

    @pytest.mark.integration
    def test_cross_entropy_gradient(self, use_gems):
        """测试CrossEntropy梯度"""
        logits = torch.randn(8, 10, device=device, requires_grad=True)
        target = torch.randint(0, 10, (8,), device=device)

        ce = nn.CrossEntropyLoss()
        loss = ce(logits, target)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestMSELoss:
    """MSE损失测试"""

    @pytest.mark.integration
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_mse_reduction_modes(self, reduction, use_gems):
        """测试MSE不同归约模式"""
        pred = torch.randn(16, 64, device=device)
        target = torch.randn(16, 64, device=device)

        mse = nn.MSELoss(reduction=reduction)
        loss = mse(pred, target)

        if reduction == "none":
            assert loss.shape == pred.shape
        else:
            assert loss.dim() == 0  # 标量
            assert loss >= 0

            # Reference comparison for scalar losses
            ref_pred = pred.detach().to("cpu").to(torch.float64)
            ref_target = target.detach().to("cpu").to(torch.float64)
            ref_loss = nn.MSELoss(reduction=reduction)(ref_pred, ref_target).to(device)
            assert_loss_close(loss, ref_loss, pred.dtype, name=f"MSE {reduction}")

    @pytest.mark.integration
    def test_mse_gradient_flow(self, use_gems):
        """测试MSE梯度流"""
        pred = torch.randn(8, 32, device=device, requires_grad=True)
        target = torch.randn(8, 32, device=device)

        mse = nn.MSELoss()
        loss = mse(pred, target)
        loss.backward()

        assert pred.grad is not None
        # 梯度应等于 2*(pred-target)/n (mean模式)
        expected_grad = 2 * (pred - target) / pred.numel()
        assert torch.allclose(pred.grad, expected_grad, rtol=1e-3)


class TestMarginRankingLoss:
    """Margin Ranking Loss测试"""

    @pytest.mark.integration
    def test_margin_ranking_basic(self, use_gems):
        """测试基本功能"""
        anchor = torch.randn(8, device=device)
        positive = torch.randn(8, device=device)
        _ = torch.randn(8, device=device)  # negative unused in this test

        # 正样本应该比负样本得分高
        target = torch.ones(8, device=device)  # anchor应该 > positive

        loss_fn = nn.MarginRankingLoss(margin=1.0)
        loss = loss_fn(anchor, positive, target)

        assert torch.isfinite(loss)

    @pytest.mark.integration
    def test_margin_satisfaction(self, use_gems):
        """测试margin满足情况"""
        # 明显满足margin的情况
        anchor = torch.tensor([2.0, 3.0, 4.0], device=device)
        positive = torch.tensor([0.0, 0.0, 0.0], device=device)
        target = torch.ones(3, device=device)

        loss_fn = nn.MarginRankingLoss(margin=1.0)
        loss = loss_fn(anchor, positive, target)

        # 当anchor > positive + margin时，loss应为0（接近）
        assert loss < 0.1


class TestMultiTaskLossCombination:
    """多任务损失组合测试"""

    @pytest.mark.integration
    def test_multi_task_loss_weighted(self, use_gems):
        """测试加权多任务损失"""
        model = MultiTaskLoss(weights=[1.0, 2.0, 0.5])

        # 模拟多个任务的输出
        logits_cls = torch.randn(4, 10, device=device)
        target_cls = torch.randint(0, 10, (4,), device=device)
        pred_reg = torch.randn(4, 64, device=device)
        target_reg = torch.randn(4, 64, device=device)
        anchor = torch.randn(4, device=device)
        positive = torch.randn(4, device=device)
        label = torch.ones(4, device=device)

        total_loss = model(
            logits_cls, target_cls, pred_reg, target_reg, anchor, positive, label
        )

        assert torch.isfinite(total_loss)
        assert total_loss >= 0

    @pytest.mark.integration
    def test_multi_task_gradient_flow(self, use_gems):
        """测试多任务损失的梯度流"""
        model = MultiTaskLoss()

        # 需要梯度的输入
        logits_cls = torch.randn(4, 10, device=device, requires_grad=True)
        target_cls = torch.randint(0, 10, (4,), device=device)
        pred_reg = torch.randn(4, 64, device=device, requires_grad=True)
        target_reg = torch.randn(4, 64, device=device)
        anchor = torch.randn(4, device=device, requires_grad=True)
        positive = torch.randn(4, device=device, requires_grad=True)
        label = torch.ones(4, device=device)

        total_loss = model(
            logits_cls, target_cls, pred_reg, target_reg, anchor, positive, label
        )
        total_loss.backward()

        # 验证所有输入都有梯度
        assert logits_cls.grad is not None
        assert pred_reg.grad is not None
        assert anchor.grad is not None
        assert positive.grad is not None


class TestContrastiveLoss:
    """对比学习损失测试"""

    @pytest.mark.integration
    def test_contrastive_loss_basic(self, use_gems):
        """测试对比损失基本功能"""
        features = torch.randn(16, 128, device=device)
        labels = torch.randint(0, 4, (16,), device=device)  # 4个类别

        loss_fn = ContrastiveLoss(temperature=0.07)
        loss = loss_fn(features, labels)

        assert torch.isfinite(loss)
        assert loss >= 0

    @pytest.mark.integration
    @pytest.mark.numerical_stability
    def test_contrastive_loss_temperature(self, use_gems):
        """测试温度参数影响"""
        features = torch.randn(8, 64, device=device)
        labels = torch.randint(0, 2, (8,), device=device)

        # 高温度（更平滑）
        loss_fn_high = ContrastiveLoss(temperature=1.0)
        loss_high = loss_fn_high(features, labels)

        # 低温度（更尖锐）
        loss_fn_low = ContrastiveLoss(temperature=0.01)
        loss_low = loss_fn_low(features, labels)

        # 两者都应该是有限值
        assert torch.isfinite(loss_high)
        assert torch.isfinite(loss_low)


class TestLossNumericalStability:
    """损失函数数值稳定性测试"""

    @pytest.mark.integration
    @pytest.mark.numerical_stability
    def test_label_smoothing_effect(self, use_gems):
        """测试标签平滑的数值稳定性"""
        logits = torch.randn(4, 10, device=device) * 10
        target = torch.randint(0, 10, (4,), device=device)

        # 无平滑
        ce_no_smooth = nn.CrossEntropyLoss()
        loss_no_smooth = ce_no_smooth(logits, target)

        # 有平滑
        ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_smooth = ce_smooth(logits, target)

        # 平滑后的loss应该略高（更保守）
        assert torch.isfinite(loss_smooth)
        assert torch.isfinite(loss_no_smooth)

    @pytest.mark.integration
    def test_loss_with_very_small_predictions(self, use_gems):
        """测试极小预测值的稳定性"""
        pred_small = torch.randn(4, 64, device=device) * 1e-8
        target = torch.randn(4, 64, device=device)

        mse = nn.MSELoss()
        loss = mse(pred_small, target)

        assert torch.isfinite(loss)
