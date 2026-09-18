import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

_SEED = 0

# Vector norms: the full torch.linalg.vector_norm ord table (None → 2).
_VECTOR_ORDS = [None, 2, 1, 0, 0.5, 3, -1, float("inf"), float("-inf")]
# Subset whose magnitude stays fp16-representable for large reductions:
# ||x||_0.5 / ||x||_3 grow as N^2 / N^(1/3) and overflow fp16 for big N.
_VECTOR_ORDS_SMALL = [None, 2, 1, 0, -1, float("inf"), float("-inf")]

_SHAPES_1D = [(8,), (1024,), (65536,)]

_ND_VECTOR_DIMS = [None, 0, [1], (1,), -1]

# 2D matrices.  Non-SVD ords here; SVD ords (2, -2, 'nuc') get their own group.
_SHAPES_2D = [(2, 5), (3, 4), (4, 4), (16, 16), (32, 32), (2, 128), (2, 2048)]
_MATRIX_ORDS = [None, "fro", 1, -1, float("inf"), float("-inf")]

_SHAPES_BATCH = [(3, 8, 8), (4, 32, 64), (2, 3, 4, 5)]
_BATCH_MATRIX_DIMS = [(-2, -1), (0, 2)]

# SVD shapes (k=2 closed form; small/medium square for gram-tridiag path).
_SVD_SHAPES = [(2, 5), (3, 4), (16, 16), (2, 128), (2, 2048), (128, 2)]
_SVD_ORDS = [2, -2, "nuc"]

if QUICK_MODE:
    _SHAPES_1D = [(1024,)]
    _SHAPES_2D = [(3, 4), (2, 128)]
    _SHAPES_BATCH = [(3, 8, 8)]
    _SVD_SHAPES = [(2, 128)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(shape, dtype, device):
    # randn-scale inputs.  NOTE: the vector branch reuses vector_norm, whose
    # full-reduction mid buffer is allocated in the output dtype; inputs with
    # |x| ≳ 1e2 (per-block sum of squares exceeding fp16 range) overflow on
    # the dim=None path for fp16/bf16.  randn-scale data stays well below.
    g = torch.Generator(device=device)
    g.manual_seed(_SEED)
    return torch.randn(shape, dtype=dtype, generator=g, device=device)


def _is_svd(ord):
    if ord is None:
        return False
    return ord == "nuc" if isinstance(ord, str) else abs(float(ord)) == 2


def _skip_non_svd_on_ascend(ord):
    """Ascend: non-SVD ords (1/-1/inf/-inf/fro) crash/hang CANN native.
    Only SVD-based ords (2, -2, nuc) are available via pure Triton kernels."""
    if flag_gems.vendor_name == "ascend" and not _is_svd(ord):
        pytest.skip("non-SVD ord crashes CANN native on Ascend")


def _svd_ok(shape):
    k, rows = min(shape[-2], shape[-1]), max(shape[-2], shape[-1])
    return k >= 2 and k <= 512 and rows <= 2048


def _svd_dtype_ok(dtype):
    """torch does not support fp16/bf16 SVD."""
    return dtype not in (torch.float16, torch.bfloat16)


def _get_atol(dtype, ord):
    if flag_gems.vendor_name in ("metax", "thead", "hygon") and _is_svd(ord):
        return 2e-3  # same as test_svd.py
    if dtype in (torch.float64, torch.float32) and ord == 0.5:
        # (Σ|x|^0.5)² magnifies last-ulp pow differences; fp64 rtol is 1e-7,
        # and the fp32 CPU reference carries ~1.3e-6 accumulation noise.
        return 1e-2
    return 1e-4


# ===========================================================================
# 1D vectors — all vector ords
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", _VECTOR_ORDS)
@pytest.mark.parametrize("shape", _SHAPES_1D)
def test_vector_1d(dtype, ord, shape):
    if dtype == torch.float16 and ord not in _VECTOR_ORDS_SMALL:
        pytest.skip("fp16 result magnitude out of range for large reductions")
    if dtype == torch.float64 and ord == 0.5:
        # (Σ|x|^0.5)² magnifies last-ulp pow differences; fp64 rtol is 1e-7.
        pytest.skip("pow rounding exceeds fp64 rtol")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = torch.linalg.norm(utils.to_reference(A), ord, dim=None)
    res = flag_gems.linalg_norm(A, ord=ord, dim=None)
    # reduce_dim scales atol with the reduction length: the 0.5-norm squares
    # the accumulation error, so the tolerance must grow with N.
    utils.gems_assert_close(
        res, ref, dtype, reduce_dim=shape[0], atol=_get_atol(dtype, ord)
    )


# ===========================================================================
# n-D vectors — dim as int / 1-tuple / None
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", _VECTOR_ORDS_SMALL)
@pytest.mark.parametrize("dim", _ND_VECTOR_DIMS)
def test_vector_nd(dtype, ord, dim):
    if dim is None and ord is not None:
        pytest.skip("dim=None with explicit ord requires 1D/2D input")
    A = _make_input((4, 8, 16), dtype, flag_gems.device)
    ref = torch.linalg.norm(utils.to_reference(A), ord, dim=dim)
    res = flag_gems.linalg_norm(A, ord=ord, dim=dim)
    utils.gems_assert_close(res, ref, dtype, atol=_get_atol(dtype, ord))


# ===========================================================================
# 2D input, explicit numeric ord, dim=None — torch routes these to
# linalg_matrix_norm over (-2, -1) (covered by test_matrix_2d /
# test_matrix_2d_svd; ord=0/3 correctly raise "Order not supported").
# ===========================================================================


# ===========================================================================
# 2D matrices — non-SVD ords (default dim = (-2, -1))
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", _MATRIX_ORDS)
@pytest.mark.parametrize("shape", _SHAPES_2D)
def test_matrix_2d(dtype, ord, shape):
    _skip_non_svd_on_ascend(ord)
    A = _make_input(shape, dtype, flag_gems.device)
    ref = torch.linalg.norm(utils.to_reference(A), ord, dim=None)
    res = flag_gems.linalg_norm(A, ord=ord, dim=None)
    utils.gems_assert_close(res, ref, dtype, atol=_get_atol(dtype, ord))


# ===========================================================================
# 2D matrices — SVD-based ords (2 / -2 / nuc)
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", _SVD_ORDS)
@pytest.mark.parametrize("shape", _SVD_SHAPES)
def test_matrix_2d_svd(dtype, ord, shape):
    if not _svd_dtype_ok(dtype):
        pytest.skip("torch does not support fp16/bf16 SVD")
    if not _svd_ok(shape):
        pytest.skip("SVD shape out of range")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = torch.linalg.norm(utils.to_reference(A), ord, dim=None)
    res = flag_gems.linalg_norm(A, ord=ord, dim=None)
    utils.gems_assert_close(res, ref, dtype, atol=_get_atol(dtype, ord))


# ===========================================================================
# Batched matrices — dim as 2-tuple (non-SVD ords; SVD requires last-two dims)
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", [None, "fro", 1, float("inf")])
@pytest.mark.parametrize("dim", _BATCH_MATRIX_DIMS)
@pytest.mark.parametrize("shape", _SHAPES_BATCH)
def test_matrix_batch(dtype, ord, dim, shape):
    _skip_non_svd_on_ascend(ord)
    A = _make_input(shape, dtype, flag_gems.device)
    ref = torch.linalg.norm(utils.to_reference(A), ord, dim=dim)
    res = flag_gems.linalg_norm(A, ord=ord, dim=dim)
    utils.gems_assert_close(res, ref, dtype, atol=_get_atol(dtype, ord))


# ===========================================================================
# keepdim + dtype= (widen-only, matching torch's narrowing rejection)
# ===========================================================================


@pytest.mark.linalg_norm
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("case", ["vector", "matrix"])
def test_keepdim_dtype(keepdim, case):
    out_dtype = (
        torch.float32
        if flag_gems.vendor_name in ("iluvatar", "ascend")
        else torch.float64
    )
    if case == "vector":
        ord, dim = 2, 1
        A = _make_input((4, 8, 16), torch.float32, flag_gems.device)
    else:
        ord, dim = "fro", (-2, -1)
        _skip_non_svd_on_ascend(ord)
        A = _make_input((3, 4), torch.float32, flag_gems.device)
    ref = torch.linalg.norm(
        utils.to_reference(A), ord, dim=dim, keepdim=keepdim, dtype=out_dtype
    )
    res = flag_gems.linalg_norm(A, ord=ord, dim=dim, keepdim=keepdim, dtype=out_dtype)
    assert res.dtype == out_dtype, f"{res.dtype} vs {out_dtype}"
    assert res.shape == ref.shape, f"{res.shape} vs {ref.shape}"
    utils.gems_assert_close(res, ref, out_dtype, atol=1e-4)


# ===========================================================================
# Error paths
# ===========================================================================


@pytest.mark.linalg_norm
def test_bad_dim_length_rejected():
    A = torch.randn(3, 4, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_norm(A, 2, [])
    with pytest.raises(RuntimeError):
        flag_gems.linalg_norm(A, 2, [0, 1, 2])


@pytest.mark.linalg_norm
def test_ord_without_dim_requires_1d_2d():
    A = torch.randn(2, 4, 5, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_norm(A, 2)


@pytest.mark.linalg_norm
def test_unsupported_matrix_ord_rejected():
    A = torch.randn(3, 4, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_norm(A, 3)


@pytest.mark.linalg_norm
def test_fro_on_1d_rejected():
    # ord='fro' routes to linalg_matrix_norm, which requires ndim >= 2
    A = torch.randn(5, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_norm(A, "fro")
