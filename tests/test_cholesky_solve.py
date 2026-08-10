import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

VENDOR_NAME = getattr(flag_gems, "vendor_name", "")
IS_ASCEND = VENDOR_NAME == "ascend"
IS_THEAD = VENDOR_NAME == "thead"

if IS_ASCEND:
    from flag_gems.runtime.backend._ascend.ops.cholesky_solve import (
        cholesky_solve,
        cholesky_solve_out,
    )
else:
    from flag_gems.ops.cholesky_solve import cholesky_solve, cholesky_solve_out


CHOLESKY_SOLVE_BASIC_SHAPES = [
    (2, 1),
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 8),
    (32, 16),
]
CHOLESKY_SOLVE_LARGE_SHAPES = [(64, 8), (128, 4)]
CHOLESKY_SOLVE_BLOCKED_SINGLE_RHS_SHAPES = [
    (8, 1),
    (16, 1),
    (32, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    (2, 128, 1),
]
CHOLESKY_SOLVE_SMALL_GATHER_SINGLE_RHS_SHAPES = [(33, 1), (48, 1), (63, 1)]
CHOLESKY_SOLVE_FP64_BLOCKED_SHAPES = [
    (64, 4),
    (128, 16),
    (256, 16),
    (256, 128),
]
CHOLESKY_SOLVE_FP32_BLOCKED_UPPER_SHAPES = [
    (64, 4),
    (64, 33),
    (64, 128),
    (128, 16),
    (128, 64),
    (256, 128),
]
CHOLESKY_SOLVE_FP32_BLOCKED_LOWER_SHAPES = [
    (128, 16),
    (128, 64),
    (256, 16),
    (256, 128),
]
CHOLESKY_SOLVE_BATCH_SHAPES = [(2, 4, 1), (3, 8, 2), (2, 3, 16, 4)]
CHOLESKY_SOLVE_RHS_BOUNDARY_SHAPES = [
    (64, 15),
    (64, 16),
    (64, 17),
    (64, 31),
    (64, 32),
    (64, 33),
]
CHOLESKY_SOLVE_UPPER_SHAPES = [
    (2, 1),
    (4, 2),
    (8, 4),
    (16, 8),
    (2, 16, 4),
    (8, 32, 8),
]
CHOLESKY_SOLVE_BROADCAST_SHAPES = [
    ((2, 1, 3, 4, 4), (2, 1, 3, 4, 6)),
    ((2, 1, 3, 4, 4), (4, 6)),
    ((4, 4), (2, 1, 3, 4, 2)),
    ((1, 3, 1, 4, 4), (2, 1, 3, 4, 5)),
]
CHOLESKY_SOLVE_COMPLEX_SHAPES = [
    (2, 1),
    (4, 2),
    (16, 5),
    (33, 4),
    (64, 8),
    (2, 16, 4),
]
CHOLESKY_SOLVE_COMPLEX_BLOCKED_SHAPES = [
    (64, 1),
    (64, 4),
    (64, 33),
    (128, 1),
    (128, 16),
    (256, 1),
    (256, 128),
]


def _use_cpu_complex_path(dtype):
    return IS_THEAD and dtype in (torch.complex64, torch.complex128)


def _move_setup_tensors_to_device(build_on_cpu, *tensors):
    if build_on_cpu:
        return tuple(tensor.to(flag_gems.device) for tensor in tensors)
    return tensors


def _make_cholesky_solve_inputs(shape, dtype, matrix_scale=1.0, rhs_scale=1.0):
    *batch_dims, n, nrhs = shape
    build_on_cpu = _use_cpu_complex_path(dtype)
    setup_device = "cpu" if build_on_cpu else flag_gems.device
    B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=setup_device)
    eye = torch.eye(n, dtype=dtype, device=setup_device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    A = matrix_scale * (B_mat @ B_mat.mH) + eye * 0.5
    L = torch.linalg.cholesky(A)
    rhs = rhs_scale * torch.randn(
        *batch_dims, n, nrhs, dtype=dtype, device=setup_device
    )
    return _move_setup_tensors_to_device(build_on_cpu, A, L, rhs)


def _make_cholesky_solve_broadcast_inputs(A_shape, rhs_shape, dtype, upper=False):
    *batch_dims, n, _ = A_shape
    build_on_cpu = _use_cpu_complex_path(dtype)
    setup_device = "cpu" if build_on_cpu else flag_gems.device
    B_mat = torch.randn(*batch_dims, n, n, dtype=dtype, device=setup_device)
    eye = torch.eye(n, dtype=dtype, device=setup_device)
    for _ in batch_dims:
        eye = eye.unsqueeze(0)
    A = B_mat @ B_mat.mH + eye * 0.5
    factor = torch.linalg.cholesky(A, upper=upper)
    rhs = torch.randn(*rhs_shape, dtype=dtype, device=setup_device)
    return _move_setup_tensors_to_device(build_on_cpu, A, factor, rhs)


def _make_conditioned_inputs(shape, dtype):
    *batch_dims, n, nrhs = shape
    build_on_cpu = _use_cpu_complex_path(dtype)
    setup_device = "cpu" if build_on_cpu else flag_gems.device
    is_single_precision = dtype in (torch.float32, torch.complex64)
    real_dtype = torch.float32 if is_single_precision else torch.float64
    condition = 1e3 if is_single_precision else 1e6
    Q_src = torch.randn(*batch_dims, n, n, dtype=dtype, device=setup_device)
    Q, _ = torch.linalg.qr(Q_src)
    eigs = torch.logspace(
        0.0,
        math.log10(condition),
        n,
        dtype=real_dtype,
        device=setup_device,
    )
    A = (Q * eigs) @ Q.mH
    L = torch.linalg.cholesky(A)
    rhs = torch.randn(*batch_dims, n, nrhs, dtype=dtype, device=setup_device)
    return _move_setup_tensors_to_device(build_on_cpu, A, L, rhs)


def _make_noncontiguous_last_dim(tensor):
    holder = torch.empty(
        *tensor.shape[:-1],
        tensor.shape[-1] * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    holder[..., ::2] = tensor
    return holder[..., ::2]


def _reference_cholesky_solve(rhs, factor, upper=False):
    if _use_cpu_complex_path(rhs.dtype):
        ref_out = torch.cholesky_solve(
            rhs.detach().cpu(), factor.detach().cpu(), upper=upper
        )
        return ref_out if utils.TO_CPU else ref_out.to(rhs.device)

    ref_rhs = utils.to_reference(rhs)
    ref_factor = utils.to_reference(factor)
    return torch.cholesky_solve(ref_rhs, ref_factor, upper=upper)


def _solve_with_gems(rhs, L, upper=False):
    if IS_ASCEND:
        return cholesky_solve(rhs, L, upper=upper)

    with flag_gems.use_gems(include=["cholesky_solve"]):
        assert "cholesky_solve" in flag_gems.current_work_registrar.get_all_keys()
        return torch.cholesky_solve(rhs, L, upper=upper)


def _solve_out_with_gems(rhs, L, out, upper=False):
    if IS_ASCEND:
        return cholesky_solve_out(rhs, L, upper=upper, out=out)

    with flag_gems.use_gems(include=["cholesky_solve", "cholesky_solve_out"]):
        registered_keys = flag_gems.current_work_registrar.get_all_keys()
        assert "cholesky_solve" in registered_keys
        assert "cholesky_solve.out" in registered_keys
        return torch.cholesky_solve(rhs, L, upper=upper, out=out)


def _assert_cholesky_solve_close(result, reference, dtype):
    if dtype == torch.complex128:
        result = utils.to_cpu(result, reference)
        torch.testing.assert_close(result, reference, atol=1e-7, rtol=1e-7)
    else:
        utils.gems_assert_close(result, reference, dtype)


def _assert_backward_error(A, X, rhs, dtype):
    if IS_ASCEND or _use_cpu_complex_path(dtype):
        # Ascend falls back to CPU during input construction, while T-Head does
        # not support the complex GEMM used by the residual calculation.
        A = A.detach().contiguous().cpu()
        X = X.detach().contiguous().cpu()
        rhs = rhs.detach().contiguous().cpu()

    residual = A @ X - rhs
    denom = A.norm() * X.norm() + rhs.norm()
    is_single_precision = dtype in (torch.float32, torch.complex64)
    real_dtype = torch.float32 if is_single_precision else torch.float64
    backward_error = residual.norm() / denom.clamp_min(torch.finfo(real_dtype).eps)
    threshold = 1e-3 if is_single_precision else 1e-10
    assert (
        backward_error.item() < threshold
    ), f"Backward error too large: {backward_error.item()} >= {threshold}"


def _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=False):
    ref_out = _reference_cholesky_solve(rhs, factor, upper=upper)
    res_out = _solve_with_gems(rhs, factor, upper=upper)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    _assert_backward_error(A, res_out, rhs.expand_as(res_out), dtype)


_REAL_DTYPES = [torch.float32] if IS_ASCEND else [torch.float32, torch.float64]
_COMPLEX_DTYPES = [] if IS_ASCEND else [torch.complex64, torch.complex128]


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_BASIC_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("contiguous_factor", [False, True])
def test_cholesky_solve(shape, dtype, contiguous_factor):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    if contiguous_factor:
        L = L.contiguous()
    ref_out = _reference_cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_LARGE_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_larger_shapes(shape, dtype):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    ref_out = _reference_cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_RHS_BOUNDARY_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_rhs_boundaries(shape, dtype):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    _assert_cholesky_solve_matches(A, L, rhs, dtype, upper=False)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_UPPER_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_upper(shape, dtype):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    U = L.mH.contiguous()
    ref_out = _reference_cholesky_solve(rhs, U, upper=True)
    res_out = _solve_with_gems(rhs, U, upper=True)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_transpose_contiguous_factor(upper):
    dtype = torch.float32
    A, L, rhs = _make_cholesky_solve_inputs((64, 16), dtype)
    factor_c = L.mT.contiguous() if upper else L.contiguous()
    factor = factor_c.mT.contiguous().mT

    assert not factor.is_contiguous()
    assert factor.mT.is_contiguous()
    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_BLOCKED_SINGLE_RHS_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_blocked_single_rhs(shape, dtype, upper):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="GPU small-gather dispatch")
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_SMALL_GATHER_SINGLE_RHS_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_small_gather_single_rhs(shape, dtype, upper):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific small-N layout path")
@pytest.mark.parametrize("batch_size", [64, 256])
def test_cholesky_solve_ascend_batched_small_lower_single_rhs(batch_size):
    dtype = torch.float32
    A, factor, rhs = _make_cholesky_solve_inputs((batch_size, 16, 1), dtype)

    assert not factor.is_contiguous()
    assert factor.mT.is_contiguous()
    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=False)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(not IS_ASCEND, reason="Ascend-specific blocked single-RHS path")
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_ascend_blocked_single_rhs_conditioned(upper):
    dtype = torch.float32
    A, L, rhs = _make_conditioned_inputs((128, 1), dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="fp64 not supported on Ascend")
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_FP64_BLOCKED_SHAPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_fp64_blocked(shape, upper):
    dtype = torch.float64
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_FP32_BLOCKED_UPPER_SHAPES)
def test_cholesky_solve_fp32_blocked_upper(shape):
    dtype = torch.float32
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous()

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=True)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_FP32_BLOCKED_LOWER_SHAPES)
@pytest.mark.parametrize("contiguous_factor", [False, True])
def test_cholesky_solve_fp32_blocked_lower(shape, contiguous_factor):
    dtype = torch.float32
    A, factor, rhs = _make_cholesky_solve_inputs(shape, dtype)
    if contiguous_factor:
        factor = factor.contiguous()

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=False)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("contiguous_factor", [False, True])
def test_cholesky_solve_batch(shape, dtype, contiguous_factor):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    if contiguous_factor:
        L = L.contiguous()
    ref_out = _reference_cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shapes", CHOLESKY_SOLVE_BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_broadcast_batch(shapes, dtype, upper):
    A_shape, rhs_shape = shapes
    _, L, rhs = _make_cholesky_solve_broadcast_inputs(A_shape, rhs_shape, dtype, upper)

    ref_out = _reference_cholesky_solve(rhs, L, upper=upper)
    res_out = _solve_with_gems(rhs, L, upper=upper)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    assert res_out.shape == ref_out.shape


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_noncontiguous_inputs(dtype):
    _, L, rhs = _make_cholesky_solve_inputs((16, 4), dtype)
    L_nc = _make_noncontiguous_last_dim(L)
    rhs_nc = _make_noncontiguous_last_dim(rhs)

    assert not L_nc.is_contiguous()
    assert not rhs_nc.is_contiguous()

    ref_out = _reference_cholesky_solve(rhs_nc, L_nc, upper=False)
    res_out = _solve_with_gems(rhs_nc, L_nc, upper=False)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_scaled_inputs(dtype):
    for matrix_scale, rhs_scale in [(1e-3, 1e3), (1e3, 1e-3)]:
        A, L, rhs = _make_cholesky_solve_inputs(
            (16, 4), dtype, matrix_scale=matrix_scale, rhs_scale=rhs_scale
        )
        ref_out = _reference_cholesky_solve(rhs, L, upper=False)
        res_out = _solve_with_gems(rhs, L, upper=False)

        _assert_cholesky_solve_close(res_out, ref_out, dtype)
        _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_conditioned_matrix(dtype):
    A, L, rhs = _make_conditioned_inputs((16, 4), dtype)
    ref_out = _reference_cholesky_solve(rhs, L, upper=False)
    res_out = _solve_with_gems(rhs, L, upper=False)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
def test_cholesky_solve_accuracy(dtype):
    A, L, rhs = _make_cholesky_solve_inputs((4, 2), dtype)
    X = _solve_with_gems(rhs, L, upper=False)

    _assert_backward_error(A, X, rhs, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.parametrize("shape", [(4, 2), (2, 4, 1)])
@pytest.mark.parametrize("dtype", _REAL_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_direct(shape, dtype, upper):
    _, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    ref_out = _reference_cholesky_solve(rhs, factor, upper=upper)
    res_out = cholesky_solve(rhs, factor, upper=upper)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_COMPLEX_SHAPES)
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex(shape, dtype, upper):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex_noncontiguous(dtype, upper):
    A, L, rhs = _make_cholesky_solve_inputs((2, 16, 5), dtype)
    factor = L.mH.contiguous() if upper else L
    factor = _make_noncontiguous_last_dim(factor)
    rhs = _make_noncontiguous_last_dim(rhs)

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex_broadcast(dtype, upper):
    A_shape = (2, 1, 4, 4)
    rhs_shape = (3, 4, 2)
    _, factor, rhs = _make_cholesky_solve_broadcast_inputs(
        A_shape, rhs_shape, dtype, upper
    )

    ref_out = _reference_cholesky_solve(rhs, factor, upper=upper)
    res_out = _solve_with_gems(rhs, factor, upper=upper)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    assert res_out.shape == ref_out.shape


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex_small_gather_conditioned(dtype, upper):
    A, L, rhs = _make_conditioned_inputs((16, 5), dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("shape", CHOLESKY_SOLVE_COMPLEX_BLOCKED_SHAPES)
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex_blocked(shape, dtype, upper):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_complex_blocked_conditioned(dtype, upper):
    A, L, rhs = _make_conditioned_inputs((64, 4), dtype)
    factor = L.mH.contiguous() if upper else L

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=upper)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("shape", [(16, 5), (64, 4), (128, 1), (256, 16)])
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_cholesky_solve_complex_lazy_conjugate_upper(shape, dtype):
    A, L, rhs = _make_cholesky_solve_inputs(shape, dtype)
    factor = L.mH
    assert factor.is_conj()

    _assert_cholesky_solve_matches(A, factor, rhs, dtype, upper=True)


@pytest.mark.cholesky_solve
@pytest.mark.skipif(IS_ASCEND, reason="complex not supported on Ascend")
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_cholesky_solve_complex_lazy_conjugate_with_all_gems(dtype):
    A, L, rhs = _make_cholesky_solve_inputs((64, 4), dtype)
    factor = L.mH
    assert factor.is_conj()

    ref_out = _reference_cholesky_solve(rhs, factor, upper=True)
    with flag_gems.use_gems(exclude=["zero_"]):
        res_out = torch.cholesky_solve(rhs, factor, upper=True)

    _assert_cholesky_solve_close(res_out, ref_out, dtype)
    _assert_backward_error(A, res_out, rhs, dtype)


@pytest.mark.cholesky_solve_out
@pytest.mark.parametrize("dtype", _REAL_DTYPES + _COMPLEX_DTYPES)
@pytest.mark.parametrize("upper", [False, True])
def test_cholesky_solve_out(dtype, upper):
    _, L, rhs = _make_cholesky_solve_inputs((2, 16, 4), dtype)
    factor = L.mH if upper else L
    ref_out = _reference_cholesky_solve(rhs, factor, upper=upper)
    out = torch.empty(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)
    original_data_ptr = out.data_ptr()

    res_out = _solve_out_with_gems(rhs, factor, out, upper=upper)

    assert res_out is out
    assert out.data_ptr() == original_data_ptr
    assert out.shape == ref_out.shape
    assert out.dtype == ref_out.dtype
    _assert_cholesky_solve_close(out, ref_out, dtype)


@pytest.mark.cholesky_solve_out
def test_cholesky_solve_out_resizes_for_broadcast_result():
    dtype = torch.float32
    _, factor, rhs = _make_cholesky_solve_broadcast_inputs(
        (2, 1, 4, 4), (3, 4, 2), dtype
    )
    ref_out = _reference_cholesky_solve(rhs, factor)

    empty_out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    res_out = _solve_out_with_gems(rhs, factor, empty_out)
    assert res_out is empty_out
    assert empty_out.shape == ref_out.shape
    _assert_cholesky_solve_close(empty_out, ref_out, dtype)

    nonempty_out = torch.empty(1, dtype=dtype, device=flag_gems.device)
    with pytest.warns(
        UserWarning, match="An output with one or more elements was resized"
    ):
        res_out = _solve_out_with_gems(rhs, factor, nonempty_out)
    assert res_out is nonempty_out
    assert nonempty_out.shape == ref_out.shape
    _assert_cholesky_solve_close(nonempty_out, ref_out, dtype)


@pytest.mark.cholesky_solve_out
def test_cholesky_solve_out_noncontiguous_and_alias():
    dtype = torch.float32
    _, factor, rhs = _make_cholesky_solve_inputs((2, 16, 4), dtype)
    ref_out = _reference_cholesky_solve(rhs, factor)

    holder = torch.empty(
        *ref_out.shape[:-1],
        ref_out.shape[-1] * 2,
        dtype=dtype,
        device=flag_gems.device,
    )
    out = holder[..., ::2]
    assert not out.is_contiguous()
    res_out = _solve_out_with_gems(rhs, factor, out)
    assert res_out is out
    _assert_cholesky_solve_close(out, ref_out, dtype)

    rhs_alias = rhs.clone()
    res_out = cholesky_solve_out(rhs_alias, factor, out=rhs_alias)
    assert res_out is rhs_alias
    _assert_cholesky_solve_close(rhs_alias, ref_out, dtype)


@pytest.mark.cholesky_solve_out
@pytest.mark.skipif(IS_ASCEND, reason="Ascend cholesky_solve only supports fp32")
def test_cholesky_solve_out_safe_dtype_cast():
    _, factor, rhs = _make_cholesky_solve_inputs((16, 4), torch.float32)
    ref_out = _reference_cholesky_solve(rhs, factor).to(torch.float64)
    out = torch.empty(rhs.shape, dtype=torch.float64, device=flag_gems.device)

    res_out = _solve_out_with_gems(rhs, factor, out)

    assert res_out is out
    assert out.dtype == torch.float64
    _assert_cholesky_solve_close(out, ref_out, torch.float64)


@pytest.mark.cholesky_solve_out
def test_cholesky_solve_out_rejects_incompatible_dtype():
    _, factor, rhs = _make_cholesky_solve_inputs((16, 4), torch.float32)
    out = torch.empty(rhs.shape, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="safely castable"):
        _solve_out_with_gems(rhs, factor, out)


@pytest.mark.cholesky_solve_out
def test_cholesky_solve_out_rejects_different_device():
    _, factor, rhs = _make_cholesky_solve_inputs((16, 4), torch.float32)
    out = torch.empty(rhs.shape, dtype=rhs.dtype, device="cpu")

    with pytest.raises(RuntimeError, match="same device"):
        _solve_out_with_gems(rhs, factor, out)


@pytest.mark.cholesky_solve
def test_cholesky_solve_empty_input():
    B = torch.empty(0, 0, dtype=torch.float32, device=flag_gems.device)
    L = torch.empty(0, 0, dtype=torch.float32, device=flag_gems.device)

    assert cholesky_solve(B, L) is B


@pytest.mark.cholesky_solve
def test_cholesky_solve_invalid_inputs():
    B = torch.randn(2, 1, dtype=torch.float32, device=flag_gems.device)
    L = torch.randn(2, 3, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(ValueError, match="square matrix"):
        cholesky_solve(B, L)

    B_bad_n = torch.randn(3, 1, dtype=torch.float32, device=flag_gems.device)
    L_square = torch.eye(2, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError, match="second-to-last dimension"):
        cholesky_solve(B_bad_n, L_square)

    B_bad_batch = torch.randn(3, 2, 1, dtype=torch.float32, device=flag_gems.device)
    L_bad_batch = torch.eye(2, dtype=torch.float32, device=flag_gems.device).expand(
        2, 2, 2
    )
    with pytest.raises(ValueError, match="not broadcastable"):
        cholesky_solve(B_bad_batch, L_bad_batch)

    B_bad_dtype = torch.randn(2, 1, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(AssertionError, match="same dtype"):
        cholesky_solve(B_bad_dtype, L_square)
