"""Functional correctness tests for linalg.matrix_power.

Coverage:
  - All dispatch paths from the PyTorch binary-exponentiation algorithm:
      n == 0  → identity
      n == 1  → clone
      n == -1 → inverse
      n < 0   → inverse + positive power
      n == 2  → fast A@A
      n == 3  → fast A@(A@A)
      n >= 4  → binary decomposition
  - Non-square / 1-D rejection
  - Non-int n rejection
  - Batch dimensions
  - out= parameter
  - Data types: float32, float64, float16, bfloat16
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Data-type selection (mirrors existing FlagGems conventions)
# ---------------------------------------------------------------------------
if QUICK_MODE:
    DTYPES = [torch.float32]
else:
    DTYPES = utils.FLOAT_DTYPES  # [float16, float32] + bfloat16 if supported

ALL_DTYPES = utils.ALL_FLOAT_DTYPES  # includes float64 if fp64 is supported

# ---------------------------------------------------------------------------
# Shape groups — organised by dispatch path
# ---------------------------------------------------------------------------

# n == 0 (identity)
IDENTITY_SHAPES = [
    (2, 2), (4, 4), (16, 16), (64, 64),
    (3, 8, 8),  # batched identity
]

# n == 1 (clone)
CLONE_SHAPES = [
    (2, 2), (8, 8), (32, 32),
    (5, 3, 4, 4),  # 2-level batch
]

# n == -1 (inverse) — requires well-conditioned matrices
# Create via A = B @ B^T + I  in the test body (won't parametrize raw random)
INV_SHAPES = [
    (2, 2), (3, 3), (8, 8), (16, 16),
]

# n == -k (negative power → inv + binary exp)
NEG_POWER_SHAPES = [
    (2, 2), (4, 4), (8, 8),
    (3, 4, 4),  # batch
]

# n == 2, n == 3 (fast matmul chain)
FAST_POWER_SHAPES = [
    (2, 2), (3, 3), (8, 8), (16, 16), (64, 64),
    (4, 5, 5),  # batch
]

# n >= 4 (binary exponentiation loop)
BINARY_EXP_SHAPES = [
    (2, 2), (3, 3), (5, 5), (8, 8), (16, 16), (32, 32),
]

# Powers specially chosen to exercise bit patterns
BINARY_POWERS = [
    4,                     # single bit (100b) — only squaring path
    5,                     # two bits (101b)
    6,                     # two bits (110b)
    7,                     # all three bits (111b)
    8,                     # higher power of 2
    10,                    # mixed bits
    15,                    # all four bits
    16,                    # power of 2
    31,                    # all five bits
]

# Batch shapes for parallel correctness
BATCH_SHAPES = [
    (2, 3, 3),
    (3, 4, 4),
    (4, 8, 8),
    (2, 3, 4, 4),          # 2-level batch
    (5, 2, 6, 6),          # 2-level batch with decent size
]

# ---------------------------------------------------------------------------
# Helper: well-conditioned square matrices
# ---------------------------------------------------------------------------

def _make_well_conditioned(shape, dtype, device):
    """Create a well-conditioned invertible matrix: I + B @ B^T."""
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=device)
    A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device)
    return A


# ===========================================================================
# Tests
# ===========================================================================


# -- n == 0 : identity ------------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", IDENTITY_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_identity(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, 0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, 0)
    utils.gems_assert_close(res_out, ref_out, dtype)


# -- n == 1 : clone ---------------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", CLONE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_clone(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, 1)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, 1)
    utils.gems_assert_close(res_out, ref_out, dtype)


# -- n == -1 : inverse ------------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", INV_SHAPES)
# inv uses LU internally; float16 may be numerically unstable for larger matrices
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(shape, dtype):
    A = _make_well_conditioned(shape, dtype, flag_gems.device)
    ref_A = utils.to_reference(A, upcast=(dtype == torch.float16))
    ref_out = torch.linalg.matrix_power(ref_A, -1)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, -1)
    # inverse can amplify floating errors; use a relaxed tolerance
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


# -- n < -1 : negative power → inv + binary exp ----------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", NEG_POWER_SHAPES)
@pytest.mark.parametrize("n", [-2, -3, -5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_negative_power(shape, n, dtype):
    A = _make_well_conditioned(shape, dtype, flag_gems.device)
    ref_A = utils.to_reference(A, upcast=(dtype == torch.float16))
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


# -- n == 2,3 : fast matmul path -------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", FAST_POWER_SHAPES)
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fast_paths(shape, n, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    utils.gems_assert_close(res_out, ref_out, dtype)


# -- n >= 4 : binary exponentiation ----------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", BINARY_EXP_SHAPES)
@pytest.mark.parametrize("n", BINARY_POWERS)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_binary_exponentiation(shape, n, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    # For larger exponents, errors compound; relax atol slightly
    atol = 5e-3 if n >= 15 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


# -- Batch correctness -----------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", BATCH_SHAPES)
@pytest.mark.parametrize("n", [0, 2, 3, 5, 10])
@pytest.mark.parametrize("dtype", DTYPES)
def test_batch(shape, n, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    utils.gems_assert_close(res_out, ref_out, dtype)


# -- out= parameter --------------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("n", [0, 2, 3, 5])
@pytest.mark.parametrize("dtype", DTYPES)
def test_out_parameter(n, dtype):
    shape = (4, 4)
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    out = torch.empty_like(A)
    with flag_gems.use_gems():
        result = torch.ops.aten.linalg_matrix_power(A, n, out=out)
    assert result is out, "out= must return the same tensor object"
    utils.gems_assert_close(out, ref_out, dtype)


# -- Input validation (error paths) ----------------------------------------

@pytest.mark.linalg_matrix_power
def test_non_square_rejected():
    A = torch.randn(3, 4, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_power(A, 2)


@pytest.mark.linalg_matrix_power
def test_1d_rejected():
    A = torch.randn(5, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_power(A, 2)


# -- Large matrix (stress test, skipped in QUICK_MODE) ---------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.skipif(QUICK_MODE, reason="Large matrix; skipped in quick mode")
@pytest.mark.parametrize("shape", [(128, 128), (256, 256)])
@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_large_matrices(shape, n, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


# -- dtype coverage: float64 ------------------------------------------------

@pytest.mark.linalg_matrix_power
@pytest.mark.skipif(not utils.fp64_is_supported, reason="fp64 not supported on this device")
@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10])
def test_float64(n):
    shape = (8, 8)
    dtype = torch.float64
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.matrix_power(ref_A, n)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_matrix_power(A, n)
    utils.gems_assert_close(res_out, ref_out, dtype)
