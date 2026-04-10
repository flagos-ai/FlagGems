"""
P2级测试：卷积网络组合测试

测试场景：
- ResNet基础块组合
- 残差连接组合
- 池化层组合
- 卷积反向传播
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

class ConvBNReLU(nn.Module):
    """卷积 + BatchNorm + ReLU基础块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """ResNet残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class MultiScaleBlock(nn.Module):
    """多尺度特征提取"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),
            nn.Conv2d(out_channels // 4, out_channels // 4, 5, padding=2)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)


# ========== 测试类 ==========

class TestConvolutionCombination:
    """卷积组合测试"""
    
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "batch_size,in_channels,out_channels,spatial_size",
        [
            pytest.param(1, 3, 16, 32, id="minimal"),
            pytest.param(4, 3, 64, 64, id="small"),
            pytest.param(8, 64, 128, 128, id="medium",
                        marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
            pytest.param(16, 128, 256, 224, id="large",
                        marks=pytest.mark.skipif(QUICK_MODE, reason="Quick mode")),
        ],
    )
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_conv_bn_relu_block(self, batch_size, in_channels, out_channels, spatial_size, dtype, use_gems):
        """测试Conv→BN→ReLU基础块"""
        model = ConvBNReLU(in_channels, out_channels).to(device).to(dtype)
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size, device=device, dtype=dtype)

        model.eval()
        with torch.no_grad():
            output = model(x)

        # 验证输出形状
        assert output.shape == (batch_size, out_channels, spatial_size, spatial_size)
        # 验证ReLU激活（所有值应>=0）
        assert (output >= 0).all()

        # Reference comparison (Conv+BN+ReLU ≈ 3 ops)
        ref_output = compute_reference(model, x)
        combo_assert_close(output, ref_output, x.dtype, num_ops=3, name="ConvBNReLU")
    
    @pytest.mark.integration
    @pytest.mark.parametrize("channels", [32, 64, 128])
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_residual_block_forward(self, channels, dtype, use_gems):
        """测试ResNet残差块前向传播"""
        model = ResidualBlock(channels).to(device).to(dtype)
        batch_size = 4
        spatial_size = 32

        x = torch.randn(batch_size, channels, spatial_size, spatial_size, device=device, dtype=dtype)

        model.eval()
        with torch.no_grad():
            output = model(x)

        # 验证形状
        assert output.shape == x.shape
        # 验证ReLU激活
        assert (output >= 0).all()

        # Reference comparison (ResBlock ≈ 6 ops: conv+bn+relu+conv+bn+add+relu)
        ref_output = compute_reference(model, x)
        combo_assert_close(output, ref_output, x.dtype, num_ops=6, name="ResidualBlock")
    
    @pytest.mark.integration
    def test_residual_gradient_flow(self, use_gems):
        """测试残差块梯度流"""
        channels = 64
        model = ResidualBlock(channels).to(device)
        x = torch.randn(2, channels, 32, 32, device=device, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在且有限
        assert x.grad is not None
        grad_norm = x.grad.norm().item()
        assert grad_norm > 0, "Gradient vanished"
        assert torch.isfinite(x.grad).all()
    
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "in_channels,out_channels",
        [(32, 64), (64, 128)],
    )
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_multi_scale_feature_extraction(self, in_channels, out_channels, dtype, use_gems):
        """测试多尺度特征提取"""
        model = MultiScaleBlock(in_channels, out_channels).to(device).to(dtype)
        x = torch.randn(2, in_channels, 56, 56, device=device, dtype=dtype)

        model.eval()
        with torch.no_grad():
            output = model(x)

        # 验证输出通道数
        assert output.shape[1] == out_channels
        # 验证空间尺寸保持
        assert output.shape[2:] == x.shape[2:]

        # Reference comparison (MultiScale ≈ 6 ops: 3 branches × 2 convs each)
        ref_output = compute_reference(model, x)
        combo_assert_close(output, ref_output, x.dtype, num_ops=6, name="MultiScaleBlock")


class TestPoolingCombination:
    """池化组合测试"""
    
    @pytest.mark.integration
    def test_max_pool_after_conv(self, use_gems):
        """测试卷积后的最大池化"""
        conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
        pool = nn.MaxPool2d(2, 2)
        
        x = torch.randn(4, 3, 64, 64, device=device)
        
        feature = conv(x)
        pooled = pool(feature)
        
        # 验证下采样
        assert pooled.shape == (4, 16, 32, 32)
    
    @pytest.mark.integration
    def test_avg_pool_after_conv(self, use_gems):
        """测试卷积后的平均池化"""
        conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
        
        x = torch.randn(4, 3, 64, 64, device=device)
        feature = conv(x)
        
        # 使用functional API测试avg_pool2d
        pooled = F.avg_pool2d(feature, kernel_size=2)
        
        assert pooled.shape == (4, 16, 32, 32)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("pool_type", ["max", "avg", "mixed"])
    def test_pooling_chain(self, pool_type, use_gems):
        """测试池化链"""
        x = torch.randn(2, 16, 128, 128, device=device)
        
        if pool_type == "max":
            out = F.max_pool2d(x, 2)
            out = F.max_pool2d(out, 2)
        elif pool_type == "avg":
            out = F.avg_pool2d(x, 2)
            out = F.avg_pool2d(out, 2)
        else:  # mixed
            out = F.max_pool2d(x, 2)
            out = F.avg_pool2d(out, 2)
        
        # 验证最终尺寸
        assert out.shape == (2, 16, 32, 32)


class TestConvBackward:
    """卷积反向传播测试"""
    
    @pytest.mark.integration
    def test_conv_gradient_correctness(self, use_gems):
        """测试卷积梯度正确性"""
        conv = nn.Conv2d(16, 32, 3, padding=1).to(device)
        x = torch.randn(2, 16, 32, 32, device=device, requires_grad=True)
        
        # 前向传播
        output = conv(x)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度存在
        assert x.grad is not None
        assert conv.weight.grad is not None
        assert torch.isfinite(x.grad).all()
    
    @pytest.mark.integration
    @pytest.mark.numerical_stability
    def test_conv_numerical_stability(self, use_gems):
        """测试卷积数值稳定性"""
        conv = nn.Conv2d(3, 16, 3, padding=1).to(device)
        
        # 大值输入
        x_large = torch.randn(2, 3, 32, 32, device=device) * 100
        output_large = conv(x_large)
        assert torch.isfinite(output_large).all()
        
        # 小值输入
        x_small = torch.randn(2, 3, 32, 32, device=device) * 1e-6
        output_small = conv(x_small)
        assert torch.isfinite(output_small).all()
        # 检查没有退化为零
        assert output_small.abs().max() > 1e-20


class TestConv3DCombination:
    """3D卷积组合测试"""
    
    @pytest.mark.integration
    @pytest.mark.skipif(QUICK_MODE, reason="Quick mode")
    def test_conv3d_forward(self, use_gems):
        """测试3D卷积前向传播"""
        conv3d = nn.Conv3d(16, 32, 3, padding=1).to(device)
        x = torch.randn(2, 16, 8, 32, 32, device=device)
        
        output = conv3d(x)
        
        assert output.shape == (2, 32, 8, 32, 32)
        assert torch.isfinite(output).all()
    
    @pytest.mark.integration
    def test_conv3d_gradient(self, use_gems):
        """测试3D卷积梯度"""
        conv3d = nn.Conv3d(8, 16, 3, padding=1).to(device)
        x = torch.randn(1, 8, 4, 16, 16, device=device, requires_grad=True)
        
        output = conv3d(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()