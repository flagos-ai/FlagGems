"""
Accuracy tests for Easy-difficulty operators:
  log10, logaddexp, cosh, gcd, tril, roll, leaky_relu, asinh

Coverage:
  - Small / regular / large shapes
  - 1-D through 5-D tensors (where applicable)
  - float16, float32, bfloat16 dtypes
  - In-place and out= variants
  - Special values (0, -0, inf, -inf, nan)
  - Empty tensors
  - Non-contiguous tensors
  - Integer dtypes (gcd)
  - Boundary parameter values
"""

import pytest
import torch

import flag_gems

from flag_gems.ops.log10 import log10, log10_, log10_out
from flag_gems.ops.logaddexp import logaddexp
from flag_gems.ops.cosh import cosh, cosh_, cosh_out
from flag_gems.ops.gcd import gcd
from flag_gems.ops.roll import roll
from flag_gems.ops.leaky_relu import leaky_relu, leaky_relu_, leaky_relu_backward

# Tril and asinh live in the Track1 ops folder during development
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ops.tril import tril, tril_
from flag_gems.ops.asinh import asinh, asinh_, asinh_out

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int32, torch.int64]

SMALL_SHAPES = [(1,), (1, 1), (8, 8)]
REGULAR_SHAPES = [(64, 64), (256, 256), (32, 128)]
LARGE_SHAPES = [(1024, 1024), (4096,)]
ALL_SHAPES = SMALL_SHAPES + REGULAR_SHAPES + LARGE_SHAPES

DEVICE = flag_gems.device


def _close(a, b, dtype, equal_nan=False):
    """Assert tensors are close, upcast to float32 for comparison."""
    a32 = a.float()
    b32 = b.float()
    atol = {
        torch.float16: 1e-3,
        torch.bfloat16: 0.016,
        torch.float32: 1.3e-6,
    }.get(dtype, 1e-5)
    assert torch.allclose(a32, b32, rtol=1e-4, atol=atol, equal_nan=equal_nan), (
        f"Max diff: {(a32 - b32).abs().max().item()}"
    )


# ===========================================================================
# log10
# ===========================================================================
class TestLog10:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dtype):
        x = torch.rand(shape, dtype=dtype, device=DEVICE) + 1e-3
        ref = torch.log10(x.float())
        res = log10(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inplace(self, dtype):
        x = torch.rand(64, 64, dtype=dtype, device=DEVICE) + 1e-3
        ref = torch.log10(x.float())
        log10_(x)
        _close(x, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        x = torch.rand(64, 64, dtype=dtype, device=DEVICE) + 1e-3
        out = torch.empty_like(x)
        ref = torch.log10(x.float())
        log10_out(x, out=out)
        _close(out, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        x = torch.tensor(
            [0.0, 1.0, 10.0, 100.0, float("inf"), float("nan")],
            dtype=dtype, device=DEVICE,
        )
        ref = torch.log10(x.float())
        res = log10(x)
        _close(res, ref, dtype, equal_nan=True)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_empty(self, dtype):
        for shape in [(0,), (4, 0), (2, 0, 3)]:
            x = torch.empty(shape, dtype=dtype, device=DEVICE)
            res = log10(x)
            assert res.shape == x.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_noncontiguous(self, dtype):
        x = torch.rand(64, 64, dtype=dtype, device=DEVICE).t()
        ref = torch.log10(x.float())
        res = log10(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_int_promotes_to_float(self, dtype):
        x = torch.randint(1, 100, (64, 64), dtype=dtype, device=DEVICE)
        ref = torch.log10(x.float())
        res = log10(x)
        _close(res, ref, torch.float32)

    @pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 4, 5)])
    def test_4d(self, shape):
        x = torch.rand(shape, dtype=torch.float32, device=DEVICE) + 1e-3
        ref = torch.log10(x)
        res = log10(x)
        _close(res, ref, torch.float32)


# ===========================================================================
# logaddexp
# ===========================================================================
class TestLogaddexp:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dtype):
        a = torch.randn(shape, dtype=dtype, device=DEVICE)
        b = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.logaddexp(a.float(), b.float())
        res = logaddexp(a, b)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        a = torch.tensor([float("-inf"), 0.0, float("inf"), float("nan")], dtype=dtype, device=DEVICE)
        b = torch.tensor([0.0, float("-inf"), float("inf"), 0.0], dtype=dtype, device=DEVICE)
        ref = torch.logaddexp(a.float(), b.float())
        res = logaddexp(a, b)
        _close(res, ref, dtype, equal_nan=True)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast(self, dtype):
        a = torch.randn(256, 1, dtype=dtype, device=DEVICE)
        b = torch.randn(1, 256, dtype=dtype, device=DEVICE)
        ref = torch.logaddexp(a.float(), b.float())
        res = logaddexp(a, b)
        _close(res, ref, dtype)


# ===========================================================================
# cosh
# ===========================================================================
class TestCosh:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.cosh(x.float())
        res = cosh(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inplace(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        ref = torch.cosh(x.float())
        cosh_(x)
        _close(x, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        out = torch.empty_like(x)
        ref = torch.cosh(x.float())
        cosh_out(x, out=out)
        _close(out, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        x = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
            dtype=dtype, device=DEVICE,
        )
        ref = torch.cosh(x.float())
        res = cosh(x)
        _close(res, ref, dtype, equal_nan=True)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_empty(self, dtype):
        for shape in [(0,), (4, 0)]:
            x = torch.empty(shape, dtype=dtype, device=DEVICE)
            res = cosh(x)
            assert res.shape == x.shape

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_even_property(self, dtype):
        """cosh(-x) == cosh(x)"""
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        pos = cosh(x)
        neg = cosh(-x)
        _close(pos, neg, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_noncontiguous(self, dtype):
        x = torch.randn(32, 64, dtype=dtype, device=DEVICE).t()
        ref = torch.cosh(x.float())
        res = cosh(x)
        _close(res, ref, dtype)


# ===========================================================================
# gcd
# ===========================================================================
class TestGcd:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_forward(self, shape, dtype):
        a = torch.randint(1, 1000, shape, dtype=dtype, device=DEVICE)
        b = torch.randint(1, 1000, shape, dtype=dtype, device=DEVICE)
        ref = torch.gcd(a, b)
        res = gcd(a, b)
        assert torch.equal(res, ref), f"Max diff: {(res - ref).abs().max()}"

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_zeros(self, dtype):
        a = torch.zeros(64, dtype=dtype, device=DEVICE)
        b = torch.randint(1, 100, (64,), dtype=dtype, device=DEVICE)
        ref = torch.gcd(a, b)
        res = gcd(a, b)
        assert torch.equal(res, ref)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_negative_inputs(self, dtype):
        a = torch.tensor([-12, -8, 0, 15], dtype=dtype, device=DEVICE)
        b = torch.tensor([8, -6, 5, -10], dtype=dtype, device=DEVICE)
        ref = torch.gcd(a, b)
        res = gcd(a, b)
        assert torch.equal(res, ref)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_coprime(self, dtype):
        a = torch.tensor([7, 11, 13], dtype=dtype, device=DEVICE)
        b = torch.tensor([9, 14, 17], dtype=dtype, device=DEVICE)
        ref = torch.gcd(a, b)
        res = gcd(a, b)
        assert torch.equal(res, ref)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_large_values(self, dtype):
        a = torch.tensor([2 ** 20, 2 ** 30], dtype=dtype, device=DEVICE)
        b = torch.tensor([2 ** 10, 2 ** 15], dtype=dtype, device=DEVICE)
        ref = torch.gcd(a, b)
        res = gcd(a, b)
        assert torch.equal(res, ref)


# ===========================================================================
# tril
# ===========================================================================
class TestTril:
    @pytest.mark.parametrize("shape", [(8, 8), (64, 64), (256, 256), (1024, 1024)])
    @pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, shape, diagonal, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.tril(x, diagonal)
        res = tril(x, diagonal)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_3d_batch(self, dtype):
        x = torch.randn(4, 32, 32, dtype=dtype, device=DEVICE)
        ref = torch.tril(x)
        res = tril(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_4d_batch(self, dtype):
        x = torch.randn(2, 3, 16, 16, dtype=dtype, device=DEVICE)
        ref = torch.tril(x)
        res = tril(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inplace(self, dtype):
        x = torch.randn(32, 32, dtype=dtype, device=DEVICE)
        ref = torch.tril(x.clone())
        tril_(x)
        _close(x, ref, dtype)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer(self, dtype):
        x = torch.randint(0, 100, (32, 32), dtype=dtype, device=DEVICE)
        ref = torch.tril(x)
        res = tril(x)
        assert torch.equal(res, ref)

    def test_non_square(self):
        x = torch.randn(16, 32, dtype=torch.float32, device=DEVICE)
        ref = torch.tril(x)
        res = tril(x)
        _close(res, ref, torch.float32)

    def test_1x1(self):
        x = torch.randn(1, 1, dtype=torch.float32, device=DEVICE)
        ref = torch.tril(x)
        res = tril(x)
        _close(res, ref, torch.float32)


# ===========================================================================
# roll
# ===========================================================================
class TestRoll:
    @pytest.mark.parametrize("shape,shifts,dims", [
        ((64,), 10, 0),
        ((64,), -10, 0),
        ((32, 32), [5, -3], [0, 1]),
        ((4, 8, 16), 3, 1),
        ((2, 3, 4, 5), [1, 2], [0, 3]),
        ((2, 3, 4, 5, 6), 2, 2),
    ])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, shifts, dims, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.roll(x, shifts, dims)
        res = roll(x, shifts, dims)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_no_dim_flatten(self, dtype):
        """roll without dims flattens, rolls, reshapes."""
        x = torch.randn(4, 8, dtype=dtype, device=DEVICE)
        ref = torch.roll(x, 5)
        res = roll(x, 5)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_zero_shift(self, dtype):
        x = torch.randn(32, 32, dtype=dtype, device=DEVICE)
        res = roll(x, 0, 0)
        _close(res, x, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_rotation(self, dtype):
        """Shifting by the full size should return the original."""
        x = torch.randn(32, dtype=dtype, device=DEVICE)
        res = roll(x, 32, 0)
        _close(res, x, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large(self, dtype):
        x = torch.randn(1024, 1024, dtype=dtype, device=DEVICE)
        ref = torch.roll(x, [100, -200], [0, 1])
        res = roll(x, [100, -200], [0, 1])
        _close(res, ref, dtype)


# ===========================================================================
# leaky_relu
# ===========================================================================
class TestLeakyRelu:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("slope", [0.01, 0.1, 0.5, 0.0])
    def test_forward(self, shape, dtype, slope):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.nn.functional.leaky_relu(x.float(), negative_slope=slope)
        res = leaky_relu(x, negative_slope=slope)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inplace(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        ref = torch.nn.functional.leaky_relu(x.float(), negative_slope=0.01)
        leaky_relu_(x, negative_slope=0.01)
        _close(x, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_backward(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        grad = torch.ones_like(x)
        slope = 0.01
        ref_grad = torch.where(x.float() >= 0, grad.float(), grad.float() * slope)
        res_grad = leaky_relu_backward(grad, x, negative_slope=slope)
        _close(res_grad, ref_grad, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        x = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf")],
                         dtype=dtype, device=DEVICE)
        ref = torch.nn.functional.leaky_relu(x.float(), negative_slope=0.01)
        res = leaky_relu(x)
        _close(res, ref, dtype)


# ===========================================================================
# asinh
# ===========================================================================
class TestAsinh:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_forward(self, shape, dtype):
        x = torch.randn(shape, dtype=dtype, device=DEVICE)
        ref = torch.asinh(x.float())
        res = asinh(x)
        _close(res, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_inplace(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        ref = torch.asinh(x.float())
        asinh_(x)
        _close(x, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_out(self, dtype):
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        out = torch.empty_like(x)
        ref = torch.asinh(x.float())
        asinh_out(x, out=out)
        _close(out, ref, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_special_values(self, dtype):
        x = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
            dtype=dtype, device=DEVICE,
        )
        ref = torch.asinh(x.float())
        res = asinh(x)
        _close(res, ref, dtype, equal_nan=True)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_odd_property(self, dtype):
        """asinh(-x) == -asinh(x)"""
        x = torch.randn(64, 64, dtype=dtype, device=DEVICE)
        pos = asinh(x).float()
        neg = asinh(-x).float()
        assert torch.allclose(pos, -neg, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_large_values(self, dtype):
        x = torch.tensor([1e3, -1e3, 1e6], dtype=dtype, device=DEVICE)
        ref = torch.asinh(x.float())
        res = asinh(x)
        _close(res, ref, dtype)
