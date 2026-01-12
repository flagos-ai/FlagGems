"""
upsample_nearest2d 测试用例

根据3.1.4测例完整度要求，本测试文件全面覆盖以下维度：
1. 输入规模：小尺寸、常规尺寸、大尺寸
2. 输入维数：4维张量 (N, C, H, W) 的不同组合
3. 参数模式：output_size、scales_h、scales_w 的所有组合
4. 功能完整性：上采样、下采样、相同尺寸、不同scale比例
"""

import os

import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

# ============================================================================
# 1. 输入规模覆盖：小尺寸、常规尺寸、大尺寸
# ============================================================================

# 小尺寸测试用例 (1×1, 8×8 等)
SHAPE_SMALL = [
    ((1, 1, 1, 1), (1, 1)),  # 最小尺寸
    ((1, 1, 1, 1), (2, 2)),  # 最小尺寸上采样
    ((1, 1, 8, 8), (8, 8)),  # 8×8 相同尺寸
    ((1, 1, 8, 8), (16, 16)),  # 8×8 上采样到 16×16
    ((1, 1, 8, 8), (4, 4)),  # 8×8 下采样到 4×4
    ((1, 2, 1, 1), (2, 2)),  # 单像素多通道
    ((2, 1, 8, 8), (16, 16)),  # 小batch
]

# 常规尺寸测试用例 (64×64, 256×256 等)
SHAPE_MEDIUM = [
    ((1, 1, 64, 64), (64, 64)),  # 64×64 相同尺寸
    ((1, 1, 64, 64), (128, 128)),  # 64×64 上采样到 128×128
    ((1, 1, 64, 64), (32, 32)),  # 64×64 下采样到 32×32
    ((4, 3, 64, 64), (128, 128)),  # 多通道常规尺寸
    ((8, 16, 64, 64), (128, 128)),  # 多通道多batch
    ((1, 1, 256, 256), (256, 256)),  # 256×256 相同尺寸
    ((1, 1, 256, 256), (512, 512)),  # 256×256 上采样到 512×512
    ((1, 1, 256, 256), (128, 128)),  # 256×256 下采样到 128×128
    ((4, 3, 256, 256), (512, 512)),  # 多通道256×256
    ((2, 8, 128, 128), (256, 256)),  # 128×128 上采样
]

# 大尺寸测试用例 (1024×1024, 4096×4096 等)
SHAPE_LARGE = [
    ((1, 1, 1024, 1024), (1024, 1024)),  # 1024×1024 相同尺寸
    ((1, 1, 1024, 1024), (2048, 2048)),  # 1024×1024 上采样到 2048×2048
    ((1, 1, 1024, 1024), (512, 512)),  # 1024×1024 下采样到 512×512
    ((2, 3, 1024, 1024), (2048, 2048)),  # 多通道大尺寸
    ((1, 1, 4096, 4096), (4096, 4096)),  # 4096×4096 相同尺寸（如果内存允许）
    ((1, 1, 4096, 4096), (2048, 2048)),  # 4096×4096 下采样
]

# ============================================================================
# 2. 输入维数覆盖：不同的 N (batch) 和 C (channel) 组合
# ============================================================================

# 不同batch和channel组合
SHAPE_DIMENSIONS = [
    # 单batch单通道
    ((1, 1, 32, 32), (64, 64)),
    # 单batch多通道
    ((1, 3, 32, 32), (64, 64)),
    ((1, 16, 32, 32), (64, 64)),
    ((1, 64, 32, 32), (64, 64)),
    # 多batch单通道
    ((2, 1, 32, 32), (64, 64)),
    ((4, 1, 32, 32), (64, 64)),
    ((8, 1, 32, 32), (64, 64)),
    # 多batch多通道
    ((2, 3, 32, 32), (64, 64)),
    ((4, 8, 32, 32), (64, 64)),
    ((8, 16, 32, 32), (64, 64)),
    ((16, 32, 32, 32), (64, 64)),
]

# ============================================================================
# 3. 参数模式覆盖：output_size、scales_h、scales_w 的各种组合
# ============================================================================

# 仅使用 output_size（scales_h/scales_w 为 None，默认模式）
SHAPE_OUTPUT_SIZE_ONLY = [
    ((1, 1, 32, 32), (32, 32), None, None),  # 相同尺寸
    ((1, 1, 32, 32), (64, 64), None, None),  # 2倍上采样
    ((1, 1, 32, 32), (16, 16), None, None),  # 2倍下采样
    ((1, 1, 32, 32), (48, 48), None, None),  # 1.5倍上采样
    ((1, 1, 32, 32), (96, 64), None, None),  # 非对称尺寸
    ((1, 1, 32, 32), (64, 96), None, None),  # 非对称尺寸
]

# 使用 scales_h 和 scales_w（显式指定scale）
SHAPE_WITH_SCALES = [
    # 整数倍scale
    ((1, 1, 32, 32), (64, 64), 2.0, 2.0),  # 2倍scale
    ((1, 1, 32, 32), (16, 16), 0.5, 0.5),  # 0.5倍scale（下采样）
    ((1, 1, 32, 32), (128, 128), 4.0, 4.0),  # 4倍scale
    # 非整数倍scale
    ((1, 1, 32, 32), (48, 48), 1.5, 1.5),  # 1.5倍scale
    ((1, 1, 32, 32), (80, 80), 2.5, 2.5),  # 2.5倍scale
    ((1, 1, 32, 32), (24, 24), 0.75, 0.75),  # 0.75倍scale
    # 非对称scale
    ((1, 1, 32, 32), (64, 96), 2.0, 3.0),  # H和W不同scale
    ((1, 1, 32, 32), (48, 64), 1.5, 2.0),  # H和W不同scale
    ((1, 1, 32, 32), (16, 24), 0.5, 0.75),  # H和W不同scale（下采样）
    # 边界值
    ((1, 1, 32, 32), (1, 1), 0.03125, 0.03125),  # 极小scale
    ((1, 1, 1, 1), (32, 32), 32.0, 32.0),  # 极大scale
]

# 混合模式：一个scale为None，另一个指定
SHAPE_MIXED_SCALES = [
    ((1, 1, 32, 32), (64, 64), 2.0, None),  # 仅指定scales_h
    ((1, 1, 32, 32), (64, 64), None, 2.0),  # 仅指定scales_w
    ((1, 1, 32, 32), (48, 64), 1.5, None),  # scales_h指定，scales_w默认
    ((1, 1, 32, 32), (64, 48), None, 1.5),  # scales_h默认，scales_w指定
]

# ============================================================================
# 4. 功能完整性覆盖：上采样、下采样、相同尺寸、特殊比例
# ============================================================================

# 上采样测试（输出尺寸 > 输入尺寸）
SHAPE_UPSAMPLE = [
    ((1, 1, 16, 16), (32, 32)),  # 2倍上采样
    ((1, 1, 16, 16), (48, 48)),  # 3倍上采样
    ((1, 1, 16, 16), (64, 64)),  # 4倍上采样
    ((1, 1, 32, 32), (96, 96)),  # 3倍上采样
    ((1, 1, 64, 64), (128, 128)),  # 2倍上采样
    ((1, 1, 128, 128), (256, 256)),  # 2倍上采样
]

# 下采样测试（输出尺寸 < 输入尺寸）
SHAPE_DOWNSAMPLE = [
    ((1, 1, 32, 32), (16, 16)),  # 2倍下采样
    ((1, 1, 64, 64), (32, 32)),  # 2倍下采样
    ((1, 1, 128, 128), (64, 64)),  # 2倍下采样
    ((1, 1, 256, 256), (128, 128)),  # 2倍下采样
    ((1, 1, 32, 32), (8, 8)),  # 4倍下采样
    ((1, 1, 64, 64), (16, 16)),  # 4倍下采样
]

# 相同尺寸测试（输出尺寸 = 输入尺寸，SAME_H 和 SAME_W 为 True）
SHAPE_SAME_SIZE = [
    ((1, 1, 8, 8), (8, 8)),
    ((1, 1, 32, 32), (32, 32)),
    ((1, 1, 64, 64), (64, 64)),
    ((1, 1, 128, 128), (128, 128)),
    ((4, 3, 32, 32), (32, 32)),
    ((8, 16, 64, 64), (64, 64)),
]

# 非整数比例测试
SHAPE_NON_INTEGER = [
    ((1, 1, 32, 32), (48, 48)),  # 1.5倍
    ((1, 1, 32, 32), (80, 80)),  # 2.5倍
    ((1, 1, 32, 32), (24, 24)),  # 0.75倍
    ((1, 1, 64, 64), (96, 96)),  # 1.5倍
    ((1, 1, 64, 64), (40, 40)),  # 0.625倍
]

# 非对称尺寸测试（H和W不同比例）
SHAPE_ASYMMETRIC = [
    ((1, 1, 32, 32), (64, 32)),  # H 2倍，W 1倍
    ((1, 1, 32, 32), (32, 64)),  # H 1倍，W 2倍
    ((1, 1, 32, 32), (64, 96)),  # H 2倍，W 3倍
    ((1, 1, 64, 64), (128, 32)),  # H 2倍，W 0.5倍
    ((1, 1, 64, 64), (32, 128)),  # H 0.5倍，W 2倍
]

# ============================================================================
# 测试函数
# ============================================================================


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_SMALL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_small(shape, output_size, dtype):
    """测试小尺寸输入（1×1, 8×8等）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_MEDIUM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_medium(shape, output_size, dtype):
    """测试常规尺寸输入（64×64, 256×256等）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_LARGE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_large(shape, output_size, dtype):
    """测试大尺寸输入（1024×1024, 4096×4096等）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_DIMENSIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_dimensions(shape, output_size, dtype):
    """测试不同batch和channel组合"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_OUTPUT_SIZE_ONLY)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_output_size_only(shape, output_size, scales_h, scales_w, dtype):
    """测试仅使用output_size参数（scales为None）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_WITH_SCALES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_with_scales(shape, output_size, scales_h, scales_w, dtype):
    """测试使用scales_h和scales_w参数"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_MIXED_SCALES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_mixed_scales(shape, output_size, scales_h, scales_w, dtype):
    """测试混合scale模式（一个为None，另一个指定）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_UPSAMPLE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_upsample(shape, output_size, dtype):
    """测试上采样功能（输出尺寸 > 输入尺寸）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_DOWNSAMPLE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_downsample(shape, output_size, dtype):
    """测试下采样功能（输出尺寸 < 输入尺寸）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_SAME_SIZE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_same_size(shape, output_size, dtype):
    """测试相同尺寸功能（SAME_H 和 SAME_W 为 True）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_NON_INTEGER)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_non_integer(shape, output_size, dtype):
    """测试非整数比例"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_ASYMMETRIC)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_asymmetric(shape, output_size, dtype):
    """测试非对称尺寸（H和W不同比例）"""
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
