"""
Accuracy tests for cosh (FlagGems · 赛道一 · 初级算子)

Coverage:
✅ Shapes  : 1-D / 2-D / 3-D / 4-D, small → large
✅ Dtypes  : float16 / bfloat16 / float32
✅ Boundary: 0 / inf / nan / large values
✅ Extras  : empty tensor, non-contiguous

精度标准（赛事 atol 表）:
float32  : atol=1.3e-6,  rtol=1e-4
float16  : atol=1e-3,    rtol=1e-4
bfloat16 : atol=0.016,   rtol=1e-4
"""

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.cosh import cosh

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


def _make_tensor(shape, dtype, device="cuda"):
    """Generate random tensor for cosh input."""
    return (torch.rand(shape, dtype=torch.float32, device=device) * 4 - 2).to(dtype)


def _ref(inp: torch.Tensor) -> torch.Tensor:
    """Reference using PyTorch cosh."""
    return torch.cosh(inp.float().cpu()).to(inp.dtype).to(inp.device)


# ===========================================================================
# 1. Forward
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cosh(shape, dtype):
    inp = _make_tensor(shape, dtype)
    ref = _ref(inp)
    with flag_gems.use_gems():
        res = torch.cosh(inp)
    torch.testing.assert_close(res, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 2. 特殊 / 边界值
# ===========================================================================
def test_special_values():
    """inf / nan / 0 semantics must match torch.cosh."""
    specials = torch.tensor(
        [0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0],
        dtype=torch.float32,
        device="cuda",
    )
    ref = torch.cosh(specials)
    res = cosh(specials)
    torch.testing.assert_close(res, ref, atol=1.3e-6, rtol=_RTOL, equal_nan=True)


def test_large_values():
    """Test numerical stability with large values."""
    x_large = torch.tensor([10.0, 20.0, -10.0, -20.0], dtype=torch.float32, device="cuda")
    ref_large = torch.cosh(x_large)
    res_large = cosh(x_large)
    torch.testing.assert_close(res_large, ref_large, atol=1e-2, rtol=1e-3)


def test_empty_tensor():
    """Empty tensor should not crash."""
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    res = cosh(x)
    assert res.shape == torch.Size([0])


def test_scalar_tensor():
    """0-dim scalar tensor."""
    x = torch.tensor(1.0, device="cuda")
    torch.testing.assert_close(cosh(x), torch.cosh(x), atol=1.3e-6, rtol=_RTOL)


def test_non_contiguous():
    """Non-contiguous tensor."""
    x = torch.rand(64, 128, device="cuda") * 4 - 2
    x_nc = x[::2, ::2]  # non-contiguous view
    ref = torch.cosh(x_nc.contiguous())
    res = cosh(x_nc)
    torch.testing.assert_close(res, ref, atol=1.3e-6, rtol=_RTOL)


# ===========================================================================
# 直接运行时的快速自检
# ===========================================================================
if __name__ == "__main__":
    total = len(POINTWISE_SHAPES) * len(FLOAT_DTYPES)
    print("=== 测例覆盖统计 ===")
    print(f"  shapes × dtypes = {total}")
    print("  特殊/边界测例: 5")
    print(f"  合计: {total + 5} 个测例\n")
    
    print("运行快速自检（float32 小尺寸）…")
    for shape in [(1,), (8, 8), (4, 4, 4), (2, 2, 2, 2)]:
        test_accuracy_cosh(shape, torch.float32)
    
    test_special_values()
    test_large_values()
    test_empty_tensor()
    test_scalar_tensor()
    test_non_contiguous()
    
    print("✅ 全部自检通过！")
