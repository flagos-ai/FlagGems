"""
Accuracy tests for log10 (FlagGems · 赛道一 · 初级算子)

Coverage:
  ✅ Shapes  : 1-D / 2-D / 3-D / 4-D, small → large
  ✅ Dtypes  : float16 / bfloat16 / float32
  ✅ Variants: forward / in-place (log10_) / out=
  ✅ Boundary: 0 → -inf, neg → nan, +inf → +inf, nan → nan
  ✅ Extras  : empty tensor, 0-dim scalar, non-contiguous

精度标准（赛事 atol 表）:
  float32  : atol=1.3e-6,  rtol=1e-4
  float16  : atol=1e-3,    rtol=1e-4
  bfloat16 : atol=0.016,   rtol=1e-4
"""

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.log10_ import log10, log10_out

# log10_ 在 test_accuracy_log10_ 中通过 torch.log10_ 测试，这里保留导入以保持 API 完整性
# noqa: F401
log10_ = None  # type: ignore

# ---------------------------------------------------------------------------
# 测试矩阵
# ---------------------------------------------------------------------------
POINTWISE_SHAPES = [
    # 1-D
    (1,),
    (8,),
    (64,),
    (1024,),
    (1024 * 1024,),
    # 2-D
    (1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    # 3-D
    (1, 1, 1),
    (8, 8, 8),
    (32, 64, 64),
    (128, 256, 256),
    # 4-D
    (1, 1, 1, 1),
    (2, 8, 8, 8),
    (2, 32, 64, 64),
    (2, 3, 224, 224),
]

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

_ATOL = {torch.float16: 1e-3, torch.bfloat16: 0.016, torch.float32: 1.3e-6}
_RTOL = 1e-4


def _make_positive(shape, dtype, device="cuda"):
    """生成正值张量（log10 定义域为正数）。"""
    return (torch.rand(shape, dtype=torch.float32, device=device).abs() + 1e-3).to(
        dtype
    )


def _ref(inp: torch.Tensor) -> torch.Tensor:
    """以 CPU float32 作为精度参考（天数平台 double 支持存疑，避免 CPU 端引入误差）。"""
    return torch.log10(inp.float().cpu()).to(inp.dtype).to(inp.device)


# ===========================================================================
# 1. Forward
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10(shape, dtype):
    inp = _make_positive(shape, dtype)
    ref = _ref(inp)

    with flag_gems.use_gems():
        res = torch.log10(inp)

    torch.testing.assert_close(res, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 2. In-place (log10_)
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_(shape, dtype):
    inp = _make_positive(shape, dtype)
    ref = _ref(inp)

    with flag_gems.use_gems():
        ret = torch.log10_(inp)

    assert ret is inp, "log10_ 必须返回 self（in-place 合约）"
    torch.testing.assert_close(inp, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 3. out=
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_out(shape, dtype):
    inp = _make_positive(shape, dtype)
    ref = _ref(inp)
    out = torch.empty_like(inp)

    with flag_gems.use_gems():
        ret = torch.log10(inp, out=out)

    assert ret is out, "log10.out 必须返回 out 张量"
    torch.testing.assert_close(out, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 4. 特殊 / 边界值
# ===========================================================================
def test_special_values():
    """nan / 0 / 负数 / inf 语义必须与 torch.log10 完全一致。"""
    specials = torch.tensor(
        [float("inf"), 0.0, -1.0, float("nan"), 1e-38, 1e38, 1.0, 10.0],
        dtype=torch.float32,
        device="cuda",
    )
    ref = torch.log10(specials)
    res = log10(specials)
    torch.testing.assert_close(res, ref, atol=1.3e-6, rtol=_RTOL, equal_nan=True)


def test_empty_tensor():
    """空张量不应崩溃。"""
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    res = log10(x)
    assert res.shape == torch.Size([0])


def test_scalar_tensor():
    """0 维标量张量。"""
    x = torch.tensor(100.0, device="cuda")
    torch.testing.assert_close(log10(x), torch.log10(x), atol=1.3e-6, rtol=_RTOL)


def test_non_contiguous():
    """非连续（步进）张量。"""
    x = torch.rand(64, 128, device="cuda").abs() + 1e-3
    x_nc = x[::2, ::2]  # non-contiguous view
    ref = torch.log10(x_nc.contiguous())
    res = log10(x_nc)
    torch.testing.assert_close(res, ref, atol=1.3e-6, rtol=_RTOL)


def test_out_direct():
    """直接调用 log10_out（不经由 use_gems 注册）。"""
    x = torch.rand(256, device="cuda").abs() + 1e-3
    out = torch.empty_like(x)
    log10_out(x, out=out)
    torch.testing.assert_close(out, torch.log10(x), atol=1.3e-6, rtol=_RTOL)


# ===========================================================================
# 直接运行时的快速自检
# ===========================================================================
if __name__ == "__main__":
    total = len(POINTWISE_SHAPES) * len(FLOAT_DTYPES)
    print("=== 测例覆盖统计 ===")
    print(f"  shapes × dtypes × 3 变体 = {total} × 3 = {total * 3}")
    print("  特殊/边界测例: 5")
    print(f"  合计: {total * 3 + 5} 个测例\n")

    print("运行快速自检（float32 小尺寸）…")
    for shape in [(1,), (8, 8), (4, 4, 4), (2, 2, 2, 2)]:
        test_accuracy_log10(shape, torch.float32)
        test_accuracy_log10_(shape, torch.float32)
        test_accuracy_log10_out(shape, torch.float32)
    test_special_values()
    test_empty_tensor()
    test_scalar_tensor()
    test_non_contiguous()
    test_out_direct()
    print("✅ 全部自检通过！")
