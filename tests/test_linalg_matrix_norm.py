import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ILUVATAR (CoreX) has no fp64 compute path; Ascend NPU BiSheng compiler
# rejects tl.float64 operations.  Skip fp64 tests on both backends.
if flag_gems.vendor_name in ("iluvatar", "ascend"):
    DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES
else:
    DTYPES = [torch.float32] if QUICK_MODE else (utils.FLOAT_DTYPES + [torch.float64])

_SEED = 0

# Dispatch boundary shapes + key sizes
_SHAPES_2D = [
    (8, 1),
    (1, 8),
    (2, 5),
    (2, 128),
    (128, 2),
    (2, 2048),
    (2048, 2),
    (3, 4),
    (5, 3),
    (16, 16),
    (4, 64),
    (16, 64),
    (256, 16),
    (32, 32),
    (32, 128),
    (128, 64),
    (256, 256),
    (512, 512),
    (512, 32),
    (256, 1024),
    (1024, 256),
    (2, 256),
    (256, 2),
    (8, 256),
    (16, 256),
    (256, 16),
    (1024, 64),
    (128, 1024),
    (1024, 384),
    (1024, 512),
]

_SHAPES_BATCH = [
    (3, 8, 8),
    (4, 32, 64),
    (8, 64, 128),
    (16, 2, 256),
    (2, 128, 512),
    (4, 4, 64, 64),
    (2, 5, 1),
    (8, 4, 32, 32),
    (16, 2, 16),
    (32, 16, 2),
    (32, 2, 2),
    (32, 8, 8),
    (16, 2, 2048),
]

if QUICK_MODE:
    _SHAPES_2D = [(8, 8), (4, 4), (3, 4), (8, 1), (2, 128)]
    _SHAPES_BATCH = [(3, 8, 8), (4, 32, 64), (16, 2, 2048)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(shape, dtype, device):
    g = torch.Generator(device=device)
    g.manual_seed(_SEED)
    return torch.randn(shape, dtype=dtype, generator=g, device=device)


def _is_svd(ord):
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


def _reduce_dim(shape, ord):
    return min(shape[-2], shape[-1]) if (isinstance(ord, str) and ord == "nuc") else 1


def _get_atol(dtype, ord):
    from .conftest import TO_CPU

    if flag_gems.vendor_name == "metax" and not TO_CPU and _is_svd(ord):
        return 2e-3  # same as test_svd.py
    if flag_gems.vendor_name == "thead" and _is_svd(ord):
        return 2e-3  # same as test_svd.py
    if flag_gems.vendor_name == "hygon" and _is_svd(ord):
        return 2e-3  # same as test_svd.py
    return 1e-4


def _compute_ref(A, ord, dim=(-2, -1), keepdim=False, dtype=None):
    from .conftest import TO_CPU

    ref = utils.to_reference(A)
    if _is_svd(ord):
        k = min(A.shape[-2], A.shape[-1])
        if k >= 64 or flag_gems.vendor_name in ("ascend", "iluvatar"):
            # Use CPU fp64 as gold standard for k >= 64: cuSOLVER GESDD
            # on GPU uses fp32 accumulation for fp32 inputs, so the
            # nuclear norm (sum of k singular values) accumulates k×ε
            # error.  For k=128 this is ~0.04 absolute—larger than the
            # FlagGems Triton result (which uses fp64 internally).
            # CPU LAPACK also uses fp64 internally and is the gold
            # standard for both fp32 and fp64 inputs.
            #
            # Ascend: CANN native matrix_norm for SVD-based ords goes
            # through the op-compile path (SetPrecisionMode →
            # AclSetCompileopt), which fails to initialize on CI runners
            # whose CANN python adapter cannot import the `decorator`
            # module (EC0010, ACL error 500001).  CANN native linalg is
            # unreliable on Ascend anyway (see _skip_non_svd_on_ascend),
            # so always compute the SVD reference on CPU there.
            ref = ref.cpu().double()
            result = torch.linalg.matrix_norm(ref, ord, dim, keepdim=keepdim)
            # Narrow to the requested dtype after the fp64 computation:
            # matrix_norm(..., dtype=fp32) on a fp64 input is rejected by
            # torch (narrowing is not implicit), and computing in fp64 first
            # is strictly more accurate than torch's own fp32 reference.
            if dtype is not None:
                result = result.to(dtype=dtype)
            if not TO_CPU:
                # Cast on the CPU first, then do a pure device copy: torch_npu
                # 2.x cannot handle fp64 sources in its .to() conversion kernel
                # and warns "Device do not support double dtype now, dtype cast
                # replace with float" (ToKernelNpu.cpp). Keeping the dtype cast
                # on the CPU side means no fp64 tensor ever reaches the NPU.
                result = result.to(dtype=dtype if dtype is not None else A.dtype).to(
                    A.device
                )
            return result
    # For non-SVD ords with --ref cpu: upcast to fp64 so the reference uses
    # accurate accumulation.  PyTorch CPU linalg.matrix_norm for fp32 input
    # uses fp32 summation; FlagGems uses fp64 accumulation (_use_fp64_acc)
    # on fp64-capable backends, so the GPU result is more accurate than the
    # fp32 CPU reference.  For large reductions (e.g. -inf summing 2048
    # |A[i,j]| values) the CPU fp32 error (~1.5e-6 relative) can exceed the
    # fp32 rtol and cause false-positive failures.  fro additionally has the
    # ~2e-3-after-sqrt squared-sum error.
    if TO_CPU and not _is_svd(ord):
        ref = ref.double()
        if dtype is not None and dtype != ref.dtype:
            result = torch.linalg.matrix_norm(ref, ord, dim, keepdim=keepdim)
            result = result.to(dtype=dtype)
            if not TO_CPU:
                result = result.to(device=A.device)
            return result
    return torch.linalg.matrix_norm(ref, ord, dim, keepdim=keepdim, dtype=dtype)


def _call_op(A, ord, dim=(-2, -1), keepdim=False, dtype=None):
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
def test_2d(dtype, ord, shape):
    _skip_non_svd_on_ascend(ord)
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("torch does not support fp16/bf16 SVD")
    if _is_svd(ord) and not _svd_ok(shape):
        pytest.skip("SVD shape out of range")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord)
    res = _call_op(A, ord)
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
@pytest.mark.parametrize(
    "shape", [(3, 4), (8, 8), (3, 8, 8)]
)  # (3, 8, 8): batched keepdim
def test_2d_keepdim(dtype, ord, shape, keepdim):
    _skip_non_svd_on_ascend(ord)
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("torch does not support fp16/bf16 SVD")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, keepdim=keepdim)
    res = _call_op(A, ord, keepdim=keepdim)
    assert res.shape == ref.shape, f"{res.shape} vs {ref.shape}"
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
def test_batch(dtype, ord, shape):
    _skip_non_svd_on_ascend(ord)
    dim = (-2, -1)
    if _is_svd(ord):
        if not _svd_dtype_ok(dtype):
            pytest.skip("torch does not support fp16/bf16 SVD")
        mk, mr = min(shape[-2], shape[-1]), max(shape[-2], shape[-1])
        if not (mk >= 2 and mk <= 512 and mr <= 2048):
            pytest.skip("SVD shape out of range")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=dim)
    res = _call_op(A, ord, dim=dim)
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
def test_batch_nondefault_dim(dim, ord, shape):
    _skip_non_svd_on_ascend(ord)
    dw = tuple(d % len(shape) for d in dim)
    if dw[0] == dw[1]:
        pytest.skip("identical dims")
    dtype = torch.float32
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=dim)
    res = _call_op(A, ord, dim=dim)
    utils.gems_assert_close(
        res,
        ref,
        dtype,
        reduce_dim=_reduce_dim(shape, ord) if _is_svd(ord) else 1,
        atol=_get_atol(dtype, ord),
    )


# ===========================================================================
# dtype= parameter (output dtype / upcast computation)
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("ord", [2, 1, "fro"])
def test_dtype_param(ord):
    """dtype= upcasts the computation/output.  fp64 output is exercised only
    on fp64-capable backends (ascend/iluvatar lack an fp64 compute path)."""
    _skip_non_svd_on_ascend(ord)
    A = _make_input((3, 4), torch.float32, flag_gems.device)
    out_dtype = (
        torch.float32
        if flag_gems.vendor_name in ("ascend", "iluvatar")
        else torch.float64
    )
    ref = _compute_ref(A, ord, dtype=out_dtype)
    res = _call_op(A, ord, dtype=out_dtype)
    assert res.dtype == out_dtype, f"{res.dtype} vs {out_dtype}"
    utils.gems_assert_close(
        res,
        ref,
        out_dtype,
        reduce_dim=_reduce_dim((3, 4), ord) if _is_svd(ord) else 1,
        atol=_get_atol(torch.float32, ord),
    )


# ===========================================================================
# Edge shapes
# ===========================================================================


@pytest.mark.linalg_matrix_norm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ord", [2, -2, 1, "fro", "nuc"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 8), (8, 1), (64, 1), (2, 4, 4)])
def test_edge(dtype, ord, shape):
    _skip_non_svd_on_ascend(ord)
    if _is_svd(ord) and not _svd_dtype_ok(dtype):
        pytest.skip("torch svd does not support fp16/bf16 SVD")
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, dim=(-2, -1))
    res = _call_op(A, ord, dim=(-2, -1))
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
def test_large(ord, shape):
    _skip_non_svd_on_ascend(ord)
    dtype = torch.float32
    A = _make_input(shape, dtype, flag_gems.device)
    ref = _compute_ref(A, ord, (-2, -1))
    res = _call_op(A, ord, dim=(-2, -1))
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
