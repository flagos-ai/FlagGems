import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DTYPES = [torch.float32] if QUICK_MODE else (utils.FLOAT_DTYPES + [torch.float64])
_HAS_CPP = hasattr(torch.ops.flag_gems, "linalg_matrix_norm")
_HAS_COP = hasattr(flag_gems, "c_operators") and hasattr(
    flag_gems.c_operators, "linalg_matrix_norm"
)
_API_PATHS = (
    ["py"] + (["cpp"] if _HAS_CPP else []) + (["cpp_direct"] if _HAS_COP else [])
)
_SEED = 0

# Dispatch boundary shapes + key sizes
_SHAPES_2D = [
    (8, 1),
    (1, 8),  # k=1  (fro kernel)
    (2, 64),
    (64, 2),
    (2, 5),  # k=2  (rank2 closed form)
    (2, 128),
    (128, 2),  # k=2  (rank2, benchmark core)
    (2, 2048),
    (2048, 2),  # k=2  (rank2, rows limit)
    (3, 4),
    (3, 3),
    (5, 3),  # k=3  (gesvd-only, no Jacobi)
    (4, 4),
    (8, 8),
    (16, 16),  # basic squares
    (4, 64),
    (64, 4),  # k=4  (Jacobi min)
    (16, 64),
    (64, 16),
    (256, 16),  # k=16
    (32, 32),
    (64, 32),
    (32, 128),
    (128, 32),  # k≤48 (15 sweeps)
    (64, 64),
    (128, 64),  # k>48 (20 sweeps)
    (128, 128),
    (256, 256),
    (512, 512),  # large square
    (32, 512),
    (512, 32),  # tall/wide
    (256, 1024),
    (1024, 256),  # large tall/wide
    (2, 256),
    (256, 2),  # k=2 near rows limit
    (8, 256),
    (256, 8),  # k=8 near rows limit
    (16, 256),
    (256, 16),  # k=16 near rows limit
    (64, 1024),
    (1024, 64),  # k=64 near rows limit
    (128, 1024),
    (1024, 128),  # k=128 near rows limit
    (384, 1024),
    (1024, 384),  # k=384 near rows limit
    (512, 1024),
    (1024, 512),  # k=512 rows limit
]

_SHAPES_BATCH = [
    (3, 8, 8),
    (4, 32, 64),
    (8, 64, 128),  # batched SVD
    (16, 2, 256),  # batched SVD (benchmark core)
    (2, 128, 512),  # batched SVD (benchmark comprehensive)
    (4, 4, 64, 64),  # multi-batch SVD
    (2, 5, 1),  # batched k=1
]

if QUICK_MODE:
    _SHAPES_2D = [(8, 8), (4, 4), (3, 4), (8, 1), (2, 128)]
    _SHAPES_BATCH = [(3, 8, 8), (4, 32, 64)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(shape, dtype, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(_SEED)
    return torch.randn(shape, dtype=torch.float32, generator=g).to(
        dtype=dtype, device=device
    )


def _is_svd(ord):
    return ord == "nuc" if isinstance(ord, str) else abs(float(ord)) == 2


def _svd_ok(shape):
    k, rows = min(shape[-2], shape[-1]), max(shape[-2], shape[-1])
    return k >= 2 and k <= 512 and rows <= 2048


def _svd_dtype_ok(dtype):
    """cuSOLVER does not support fp16/bf16 SVD."""
    return dtype not in (torch.float16, torch.bfloat16)


def _reduce_dim(shape, ord):
    return min(shape[-2], shape[-1]) if (isinstance(ord, str) and ord == "nuc") else 1


# CPU-mode atol for SVD ords (ord=2/-2/nuc).  FlagGems fp64 QR is ~1e-6 vs
# LAPACK, so these are intentionally tight — adjust here if needed.
_CPU_SVD_ATOL = {
    torch.float32: 1e-4,
    torch.float64: 1e-4,
}

# CPU-mode atol for fro (Frobenius norm).  After the fp64 accumulation fix
# in _fro_kernel, the GPU-CPU difference is ~3e-6 relative (dominated by
# different summation order, not precision loss).  For the largest matrices
# (norm ~1024) this is ~3e-3 absolute, so 5e-3 gives safe headroom.
_CPU_FRO_ATOL = {
    torch.float32: 1e-4,
    torch.float64: 1e-4,
}


def _svd_atol(dtype):
    """Return atol for SVD ords, aligned with test_svd.py precision requirements.
    test_svd.py uses atol=2e-3 for fp32 singular values and reconstruction.
    For nuclear norm (sum of σ_k), error scales with k via reduce_dim."""
    from .conftest import TO_CPU

    if TO_CPU:
        # CPU reference (LAPACK) — FlagGems fp64 QR achieves ~1e-6 precision.
        return _CPU_SVD_ATOL.get(dtype, 1e-4)
    if dtype == torch.float32:  # same as test_svd.py
        return 2e-3
    if dtype == torch.float64:  # same as test_svd.py
        return 2e-3
    return 1e-4


def _get_atol(dtype, ord):
    """Return atol for the given ord, aware of --ref cpu mode."""
    from .conftest import TO_CPU

    if _is_svd(ord):
        return _svd_atol(dtype)
    if TO_CPU and (isinstance(ord, str) and ord == "fro"):
        return _CPU_FRO_ATOL.get(dtype, 1e-4)
    return 1e-4


def _compute_ref(A, ord, dim=(-2, -1), keepdim=False):
    from .conftest import TO_CPU

    ref = utils.to_reference(A)
    if _is_svd(ord):
        k = min(A.shape[-2], A.shape[-1])
        if k >= 8:
            # Use CPU fp64 as gold standard for k >= 64: cuSOLVER GESDD
            # on GPU uses fp32 accumulation for fp32 inputs, so the
            # nuclear norm (sum of k singular values) accumulates k×ε
            # error.  For k=128 this is ~0.04 absolute—larger than the
            # FlagGems Triton result (which uses fp64 internally).
            # CPU LAPACK also uses fp64 internally and is the gold
            # standard for both fp32 and fp64 inputs.
            ref = ref.cpu().double()
            result = torch.linalg.matrix_norm(ref, ord, dim, keepdim=keepdim)
            if not TO_CPU:
                result = result.to(device=A.device, dtype=A.dtype)
            return result
    # For fro with --ref cpu: upcast to fp64 so the reference uses accurate
    # LAPACK accumulation.  PyTorch CPU linalg.matrix_norm for fp32 input
    # uses fp32 sequential summation, producing ~4e-3 relative error in the
    # squared sum (~2e-3 after sqrt) for large matrices.  FlagGems GPU with
    # fp64 tiled reduction is more accurate than the fp32 CPU reference,
    # which would cause false-positive test failures.
    if TO_CPU and (isinstance(ord, str) and ord == "fro"):
        ref = ref.double()
    return torch.linalg.matrix_norm(ref, ord, dim, keepdim=keepdim)


def _call_op(api_path, A, ord, dim=(-2, -1), keepdim=False, dtype=None):
    if api_path == "aten":
        with flag_gems.use_gems():
            return torch.linalg.matrix_norm(A, ord, dim, keepdim=keepdim)
    if api_path == "cpp":
        if isinstance(ord, str):
            res = torch.ops.flag_gems.linalg_matrix_norm.str_ord(
                A, ord, dim, keepdim, dtype=dtype
            )
        else:
            res = torch.ops.flag_gems.linalg_matrix_norm(A, float(ord), dim, keepdim)
        if res is None or (hasattr(res, "numel") and res.numel() == 0):
            return flag_gems.linalg_matrix_norm(
                A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype
            )
        return res
    if api_path == "cpp_direct":
        if isinstance(ord, str):
            return flag_gems.c_operators.linalg_matrix_norm_str(
                A, ord, list(dim), keepdim, dtype=dtype
            )
        else:
            return flag_gems.c_operators.linalg_matrix_norm(
                A, float(ord), list(dim), keepdim, dtype=dtype
            )
    return flag_gems.linalg_matrix_norm(
        A, ord=ord, dim=dim, keepdim=keepdim, dtype=dtype
    )


# ===========================================================================
# 2D — all ords × all shapes  (dtype outer → output grouped by dtype)
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "ord", [2, -2, 1, -1, float("inf"), float("-inf"), "fro", "nuc"]
)
@pytest.mark.parametrize("shape", _SHAPES_2D)
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_2d(dtype, ord, shape, api_path):
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("cuSOLVER does not support fp16/bf16 SVD")
    if _is_svd(ord) and not _svd_ok(shape):
        pytest.skip("SVD shape out of range")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord)
    res = _call_op(api_path, A, ord)
    # print(
    #     f"dtyp:{dtype}, {shape}, ord:{ord} ref:{ref}, res:{res}, atol:{abs(ref - res)}"
    # )
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# 2D — keepdim
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("ord", [2, 1, float("inf"), "fro", "nuc"])
@pytest.mark.parametrize("shape", [(3, 4), (8, 8)])
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_2d_keepdim(dtype, ord, shape, keepdim, api_path):
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("cuSOLVER does not support fp16/bf16 SVD")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, keepdim=keepdim)
    res = _call_op(api_path, A, ord, keepdim=keepdim)
    assert res.shape == ref.shape, f"[{api_path}] {res.shape} vs {ref.shape}"
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# Batched — all ords × dims  (default dim = (-2, -1))
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "ord", [2, -2, 1, -1, float("inf"), float("-inf"), "fro", "nuc"]
)
@pytest.mark.parametrize("shape", _SHAPES_BATCH)
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_batch(dtype, ord, shape, api_path):
    dim = (-2, -1)
    if _is_svd(ord):
        if not _svd_dtype_ok(dtype):
            pytest.skip("cuSOLVER does not support fp16/bf16 SVD")
        mk, mr = min(shape[-2], shape[-1]), max(shape[-2], shape[-1])
        if not (mk >= 2 and mk <= 512 and mr <= 2048):
            pytest.skip("SVD shape out of range")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=dim)
    res = _call_op(api_path, A, ord, dim=dim)
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# Batched — non-default dims (non-SVD ords, single dtype)
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dim", [(0, 2), (-3, -1)])
@pytest.mark.parametrize("ord", [1, float("inf"), "fro"])
@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 3, 4, 5)])
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_batch_nondefault_dim(dim, ord, shape, api_path):
    dw = tuple(d % len(shape) for d in dim)
    if dw[0] == dw[1]:
        pytest.skip("identical dims")
    dtype = torch.float32
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=dim)
    res = _call_op(api_path, A, ord, dim=dim)
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# Edge shapes
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", [2, 1, "fro"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 8), (8, 1), (64, 1), (2, 4, 4)])
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_edge(dtype, ord, shape, api_path):
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("cuSOLVER does not support fp16/bf16 SVD")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=(-2, -1))
    res = _call_op(api_path, A, ord, dim=(-2, -1))
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# Large matrix stress  (non-SVD, float32 only)
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.skipif(QUICK_MODE, reason="large matrices; skipped in quick mode")
@pytest.mark.parametrize("ord", [1, float("inf"), "fro"])
@pytest.mark.parametrize("shape", [(128, 256), (512, 512), (1024, 64)])
@pytest.mark.parametrize("api_path", _API_PATHS)
def test_large(ord, shape, api_path):
    dtype = torch.float32
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, (-2, -1))
    res = _call_op(api_path, A, ord, dim=(-2, -1))
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# Error paths
# ===========================================================================


@pytest.mark.linalg_matrix_norm
def test_1d_rejected():
    A = torch.randn(5, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_norm(A)


@pytest.mark.linalg_matrix_norm
def test_same_dim_rejected():
    A = torch.randn(3, 4, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_norm(A, 2, (0, 0))


@pytest.mark.linalg_matrix_norm
def test_unsupported_ord_rejected():
    A = torch.randn(3, 4, device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_norm(A, 3)
