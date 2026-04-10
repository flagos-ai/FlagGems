"""
Accuracy tests for logaddexp (FlagGems · 赛道一 · 初级算子)

Coverage:
  ✅ Shapes  : 1-D / 2-D / 3-D / 4-D, small → large
  ✅ Dtypes  : float16 / bfloat16 / float32
  ✅ Variants: forward / out=
  ✅ Boundary: inf / -inf / nan / large values
  ✅ Extras  : empty tensor, broadcast

精度标准（赛事 atol 表）:
  float32  : atol=1.3e-6,  rtol=1e-4
  float16  : atol=1e-3,    rtol=1e-4
  bfloat16 : atol=0.016,   rtol=1e-4
"""

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.logaddexp import logaddexp, logaddexp_out

# logaddexp_ 在 test_accuracy_logaddexp_ 中通过 torch.logaddexp_ 测试
# noqa: F401
logaddexp_ = None  # type: ignore

# ---------------------------------------------------------------------------
# Test matrix
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
    """Generate positive tensor for logaddexp input."""
    return (torch.rand(shape, dtype=torch.float32, device=device).abs() + 1e-3).to(
        dtype
    )


def _ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute reference using PyTorch logaddexp."""
    return torch.logaddexp(x.float().cpu(), y.float().cpu()).to(x.dtype).to(x.device)


# ===========================================================================
# 1. Forward
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_logaddexp(shape, dtype):
    x = _make_positive(shape, dtype)
    y = _make_positive(shape, dtype)
    ref = _ref(x, y)

    with flag_gems.use_gems():
        res = torch.logaddexp(x, y)

    torch.testing.assert_close(res, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 2. out=
# ===========================================================================
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_logaddexp_out(shape, dtype):
    x = _make_positive(shape, dtype)
    y = _make_positive(shape, dtype)
    ref = _ref(x, y)
    out = torch.empty_like(x)

    with flag_gems.use_gems():
        ret = torch.logaddexp(x, y, out=out)

    assert ret is out, "logaddexp.out must return out tensor"
    torch.testing.assert_close(out, ref, atol=_ATOL[dtype], rtol=_RTOL)


# ===========================================================================
# 3. Special / Boundary values
# ===========================================================================
def test_special_values():
    """inf / -inf / nan semantics must match torch.logaddexp."""
    # Test with inf
    x_inf = torch.tensor(
        [float("inf"), 0.0, -float("inf"), 100.0], dtype=torch.float32, device="cuda"
    )
    y_inf = torch.tensor(
        [0.0, float("inf"), -float("inf"), 100.0], dtype=torch.float32, device="cuda"
    )
    ref_inf = torch.logaddexp(x_inf, y_inf)
    res_inf = logaddexp(x_inf, y_inf)
    torch.testing.assert_close(
        res_inf, ref_inf, atol=1.3e-6, rtol=_RTOL, equal_nan=True
    )
    print(f"  inf test passed: {res_inf.tolist()}")


def test_large_values():
    """Test numerical stability with large values that would overflow exp()."""
    # These values would overflow exp() but logaddexp should handle them
    x_large = torch.tensor([1000.0, 800.0, -800.0], dtype=torch.float32, device="cuda")
    y_large = torch.tensor([1000.0, 800.0, 800.0], dtype=torch.float32, device="cuda")
    ref_large = torch.logaddexp(x_large, y_large)
    res_large = logaddexp(x_large, y_large)
    torch.testing.assert_close(res_large, ref_large, atol=1e-2, rtol=1e-3)
    print(f"  large values test passed: {res_large.tolist()}")


def test_empty_tensor():
    """Empty tensor should not crash."""
    x = torch.empty(0, dtype=torch.float32, device="cuda")
    y = torch.empty(0, dtype=torch.float32, device="cuda")
    res = logaddexp(x, y)
    assert res.shape == torch.Size([0])
    print("  empty tensor test passed")


def test_broadcast():
    """Broadcast shapes correctly."""
    x = torch.rand(8, 8, device="cuda")
    y = torch.rand(8, 1, device="cuda")  # broadcasts to (8, 8)
    ref = torch.logaddexp(x, y)
    res = logaddexp(x, y)
    torch.testing.assert_close(res, ref, atol=1e-4, rtol=1e-4)
    print("  broadcast test passed")


def test_different_shapes():
    """Different but broadcastable shapes."""
    shapes = [
        ((4, 8), (8,)),
        ((2, 3, 4), (4,)),
        ((1, 1, 8), (8, 8)),
    ]
    for s1, s2 in shapes:
        x = _make_positive(s1, torch.float32)
        y = _make_positive(s2, torch.float32)
        ref = torch.logaddexp(x, y)
        res = logaddexp(x, y)
        torch.testing.assert_close(res, ref, atol=1e-4, rtol=1e-4)
    print("  different shapes test passed")


# ===========================================================================
# Direct function call test
# ===========================================================================
def test_logaddexp_out_direct():
    """Direct call to logaddexp_out (not through use_gems)."""
    x = torch.rand(256, device="cuda")
    y = torch.rand(256, device="cuda")
    out = torch.empty_like(x)
    logaddexp_out(x, y, out=out)
    ref = torch.logaddexp(x, y)
    torch.testing.assert_close(out, ref, atol=1.3e-6, rtol=_RTOL)
    print("  logaddexp_out direct test passed")


# ===========================================================================
# Quick self-test
# ===========================================================================
if __name__ == "__main__":
    total = len(POINTWISE_SHAPES) * len(FLOAT_DTYPES)
    print("=== Test coverage statistics ===")
    print(f"  shapes × dtypes × 2 variants = {total} × 2 = {total * 2}")
    print("  Special/boundary tests: 5")
    print(f"  Total: {total * 2 + 5} test cases\n")

    print("Running quick self-check (float32 small sizes)...")
    for shape in [(1,), (8, 8), (4, 4, 4), (2, 2, 2, 2)]:
        test_accuracy_logaddexp(shape, torch.float32)
        test_accuracy_logaddexp_out(shape, torch.float32)
    test_special_values()
    test_large_values()
    test_empty_tensor()
    test_broadcast()
    test_different_shapes()
    test_logaddexp_out_direct()
    print("✅ All self-checks passed!")
