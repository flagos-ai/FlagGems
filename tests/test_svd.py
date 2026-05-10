import pytest
import torch

import flag_gems
from flag_gems.ops.svd import (
    _can_use_2x2_kernel,
    _can_use_4x4_kernel,
    _can_use_gram_eigh_kernel,
    _can_use_rank1_kernel,
    _can_use_rank2_kernel,
    _can_use_small_jacobi_kernel,
    _can_use_streaming_jacobi_kernel,
)

from . import accuracy_utils as utils
from . import conftest as cfg

FLOAT_DTYPES = [torch.float32]
if utils.fp64_is_supported:
    FLOAT_DTYPES.append(torch.float64)

LOW_PRECISION_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    LOW_PRECISION_DTYPES.append(torch.bfloat16)

COMPLEX_DTYPES = [] if cfg.QUICK_MODE else [torch.complex64]
if not cfg.QUICK_MODE and utils.fp64_is_supported:
    COMPLEX_DTYPES.append(torch.complex128)

DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES
SHAPES = [(4, 3)] if cfg.QUICK_MODE else [(1, 1), (5, 3), (3, 5), (2, 4, 4)]
TRITON_SHAPES = (
    [(8, 2, 2)]
    if cfg.QUICK_MODE
    else [
        (2, 2),
        (257, 2, 2),
        (16, 1),
        (1, 16),
        (17, 2),
        (2, 17),
        (4, 4),
        (8, 8),
        (16, 8),
        (8, 16),
        (512, 8),
        (16, 256),
        (64, 32),
        (1024, 32),
    ]
)
STREAMING_TRITON_SHAPES = (
    []
    if cfg.QUICK_MODE
    else [
        (16, 128, 64),
        (16, 128, 128),
        (16, 1024, 64),
        (16, 64, 1024),
    ]
)
GRAM_EIGH_TRITON_SHAPES = (
    []
    if cfg.QUICK_MODE
    else [
        (256, 256),
        (1024, 1024),
        (9, 9),
        (33, 33),
        (129, 129),
        (1024, 8),
        (8, 1024),
        (2, 256, 256),
        (1024, 256),
        (256, 1024),
    ]
)
LOW_PRECISION_SHAPES = (
    []
    if cfg.QUICK_MODE
    else [
        (16, 1),
        (257, 2, 2),
        (4, 4),
        (8, 8),
        (9, 9),
        (64, 32),
        (129, 129),
    ]
)


def _make_input(shape, dtype):
    if dtype.is_complex:
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        real = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
        imag = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
        return (real + 1j * imag).to(dtype)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _reconstruct(u, s, v):
    k = s.shape[-1]
    return u[..., :, :k] @ torch.diag_embed(s).to(u.dtype) @ v[..., :, :k].mH


def _assert_close(actual, expected, dtype, atol=1e-4, rtol=1e-4):
    actual = utils.to_cpu(actual, expected)
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


def _assert_orthogonal(matrix, atol=1e-4, rtol=1e-4):
    ident = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    actual = matrix.mH @ matrix
    torch.testing.assert_close(actual, ident.expand_as(actual), atol=atol, rtol=rtol)


@pytest.mark.svd
def test_svd_triton_route_covers_minmn_le_1024():
    for k in range(1, 1025):
        inp = torch.empty((k, k), dtype=torch.float32)
        has_route = (
            _can_use_2x2_kernel(inp)
            or _can_use_rank1_kernel(inp, True, True)
            or _can_use_rank2_kernel(inp, True, True)
            or _can_use_4x4_kernel(inp, True, True)
            or _can_use_small_jacobi_kernel(inp, True, True)
            or _can_use_streaming_jacobi_kernel(inp, True, True)
            or _can_use_gram_eigh_kernel(inp, True, True)
        )
        assert has_route, f"missing Triton route for square k={k}"

    for k in (1, 2, 3, 4, 5, 8, 9, 16, 32, 64, 128, 1024):
        for shape in ((2048, k), (k, 2048)):
            inp = torch.empty(shape, dtype=torch.float32)
            has_route = (
                _can_use_rank1_kernel(inp, True, True)
                or _can_use_rank2_kernel(inp, True, True)
                or _can_use_small_jacobi_kernel(inp, True, True)
                or _can_use_streaming_jacobi_kernel(inp, True, True)
                or _can_use_gram_eigh_kernel(inp, True, True)
            )
            assert has_route, f"missing Triton route for shape={shape}"


@pytest.mark.svd
@pytest.mark.parametrize("shape", TRITON_SHAPES + SHAPES)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_svd_compute_uv(shape, some, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=True)
    with flag_gems.use_gems():
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=True)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_s, ref_s, dtype, atol=5e-4, rtol=5e-4)
    _assert_close(
        _reconstruct(res_u, res_s, res_v), ref_inp, dtype, atol=5e-4, rtol=5e-4
    )
    if shape in TRITON_SHAPES and dtype == torch.float32 and some:
        _assert_orthogonal(res_u, atol=1e-3, rtol=1e-3)
        _assert_orthogonal(res_v, atol=1e-3, rtol=1e-3)


@pytest.mark.svd
@pytest.mark.parametrize("shape", LOW_PRECISION_SHAPES)
@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_svd_low_precision(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = inp.float()

    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    assert res_u.dtype == dtype
    assert res_s.dtype == dtype
    assert res_v.dtype == dtype
    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape

    atol = 8e-2 if dtype == torch.bfloat16 else 2e-2
    rtol = 8e-2 if dtype == torch.bfloat16 else 2e-2
    _assert_close(res_s.float(), ref_s, dtype, atol=atol, rtol=rtol)
    _assert_close(
        _reconstruct(res_u.float(), res_s.float(), res_v.float()),
        ref_inp,
        dtype,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.svd
@pytest.mark.parametrize("shape", TRITON_SHAPES + SHAPES)
@pytest.mark.parametrize("some", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_svd_compute_uv_false(shape, some, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=False)
    with flag_gems.use_gems():
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=False)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_u, ref_u, dtype)
    _assert_close(res_s, ref_s, dtype, atol=5e-4, rtol=5e-4)
    _assert_close(res_v, ref_v, dtype)


@pytest.mark.svd
@pytest.mark.parametrize("shape", STREAMING_TRITON_SHAPES)
def test_svd_streaming_jacobi(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_s, ref_s, torch.float32, atol=5e-4, rtol=1e-3)
    _assert_close(
        _reconstruct(res_u, res_s, res_v),
        ref_inp,
        torch.float32,
        atol=1e-3,
        rtol=1e-3,
    )
    _assert_orthogonal(res_u, atol=1e-2, rtol=1e-2)
    _assert_orthogonal(res_v, atol=1e-2, rtol=1e-2)


@pytest.mark.svd
@pytest.mark.parametrize("shape", GRAM_EIGH_TRITON_SHAPES)
def test_svd_gram_eigh(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)

    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)
    with flag_gems.use_gems():
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_s, ref_s, torch.float32, atol=3e-2, rtol=5e-3)
    _assert_close(
        _reconstruct(res_u, res_s, res_v),
        ref_inp,
        torch.float32,
        atol=2e-3,
        rtol=2e-3,
    )
    _assert_orthogonal(res_v, atol=2e-2, rtol=2e-2)


@pytest.mark.svd
def test_svd_non_contiguous_and_empty():
    inputs = [
        _make_input((3, 5), torch.float32).mT,
        torch.empty((0, 3), dtype=torch.float32, device=flag_gems.device),
        torch.empty((2, 3, 0), dtype=torch.float32, device=flag_gems.device),
    ]

    for inp in inputs:
        ref_inp = utils.to_reference(inp, False)
        ref_u, ref_s, ref_v = torch.svd(ref_inp)
        with flag_gems.use_gems():
            res_u, res_s, res_v = torch.svd(inp)

        assert res_u.shape == ref_u.shape
        assert res_s.shape == ref_s.shape
        assert res_v.shape == ref_v.shape
        _assert_close(res_s, ref_s, torch.float32)
        _assert_close(_reconstruct(res_u, res_s, res_v), ref_inp, torch.float32)
