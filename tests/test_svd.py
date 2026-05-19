import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_spectrum_input(shape, singular_values, seed=0):
    torch.manual_seed(seed)
    *batch_shape, m, n = shape
    k = min(m, n)
    left, _ = torch.linalg.qr(
        torch.randn((*batch_shape, m, m), dtype=torch.float32, device=flag_gems.device)
    )
    right, _ = torch.linalg.qr(
        torch.randn((*batch_shape, n, n), dtype=torch.float32, device=flag_gems.device)
    )
    sigma = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    diag = torch.as_tensor(
        singular_values[:k], dtype=torch.float32, device=flag_gems.device
    )
    idx = torch.arange(k, device=flag_gems.device)
    sigma[..., idx, idx] = diag
    return left @ sigma @ right.mH


def _make_input(shape, dtype):
    if dtype.is_complex:
        real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        return (real + 1j * imag).to(dtype)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _reconstruct(u, s, v):
    k = s.shape[-1]
    return u[..., :, :k] @ torch.diag_embed(s).to(u.dtype) @ v[..., :, :k].mH


def _assert_same_shape(actual, expected):
    actual_shape = torch.tensor(tuple(actual.shape))
    expected_shape = torch.tensor(tuple(expected.shape))
    utils.gems_assert_equal(actual_shape, expected_shape)


def _assert_finite(*actuals):
    for actual in actuals:
        finite = torch.isfinite(actual)
        expected = utils.to_reference(torch.ones_like(finite), False)
        utils.gems_assert_equal(finite, expected)


def _assert_orthonormal(actual, atol=2e-2):
    if actual.numel() == 0:
        return
    k = actual.shape[-1]
    eye = torch.eye(k, dtype=actual.dtype, device=actual.device)
    gram = actual.mH @ actual
    expected = utils.to_reference(eye.expand_as(gram), False)
    utils.gems_assert_close(gram, expected, gram.dtype, atol=atol)


@pytest.mark.svd
@pytest.mark.parametrize("shape", utils.SVD_FAST_SHAPES)
def test_accuracy_svd_fast_float32(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_v)


@pytest.mark.svd
@pytest.mark.parametrize("shape", utils.SVD_RANK1_SHAPES)
def test_accuracy_svd_rank1_float32(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)


@pytest.mark.svd
@pytest.mark.parametrize("shape", utils.SVD_RANK1_SHAPES)
def test_accuracy_svd_rank1_zero_basis(shape):
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    expected_s = utils.to_reference(torch.zeros_like(res_s), False)
    utils.gems_assert_equal(res_s, expected_s)
    if shape[-1] == 1:
        expected_u = utils.to_reference(torch.zeros_like(res_u), False)
        expected_v = utils.to_reference(torch.ones_like(res_v), False)
    else:
        expected_u = utils.to_reference(torch.ones_like(res_u), False)
        expected_v = utils.to_reference(torch.zeros_like(res_v), False)
    utils.gems_assert_close(res_u, expected_u, res_u.dtype)
    utils.gems_assert_close(res_v, expected_v, res_v.dtype)


@pytest.mark.svd
@pytest.mark.parametrize("shape", utils.SVD_GRAM_ILL_CONDITIONED_SHAPES)
def test_accuracy_svd_gram_ill_conditioned_orthonormal(shape):
    k = min(shape[-2:])
    singular_values = torch.logspace(0, -5, steps=k).tolist()
    inp = _make_spectrum_input(shape, singular_values, seed=7)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_v)


@pytest.mark.svd
@pytest.mark.parametrize("case", ["zero", "repeated"])
def test_accuracy_svd_gram_zero_and_repeated_singular_values(case):
    if case == "zero":
        inp = torch.zeros((17, 17), dtype=torch.float32, device=flag_gems.device)
    else:
        inp = _make_spectrum_input((17, 17), [4, 4, 2, 2, *([1] * 13)], seed=11)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_v)


@pytest.mark.svd
@pytest.mark.parametrize("case", utils.SVD_TINY_RANK_DEGENERATE_CASES)
def test_accuracy_svd_tiny_rank_degenerate_inputs(case):
    if case == "zero_2x2":
        inp = torch.zeros((2, 2), dtype=torch.float32, device=flag_gems.device)
    elif case == "repeated_2x2":
        inp = torch.diag(torch.ones(2, dtype=torch.float32, device=flag_gems.device))
    elif case == "zero_column_8x2":
        inp = torch.cat(
            [
                torch.ones((8, 1), dtype=torch.float32, device=flag_gems.device),
                torch.zeros((8, 1), dtype=torch.float32, device=flag_gems.device),
            ],
            dim=-1,
        )
    else:
        inp = torch.cat(
            [
                torch.ones((1, 8), dtype=torch.float32, device=flag_gems.device),
                torch.zeros((1, 8), dtype=torch.float32, device=flag_gems.device),
            ],
            dim=-2,
        )
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_finite(res_u, res_s, res_v)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(1025, 2), (2, 1025)])
@pytest.mark.parametrize("case", ["zero", "rankdef", "scaled"])
def test_accuracy_svd_rank2_boundary_degenerate_inputs(shape, case):
    if case == "zero":
        inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    else:
        length = max(shape)
        vec = torch.linspace(1.0, 2.0, length, device=flag_gems.device)
        small = torch.linspace(-0.5, 0.5, length, device=flag_gems.device)
        if shape[-2] >= shape[-1]:
            inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
            inp[:, 0] = vec
            inp[:, 1] = 0.0 if case == "rankdef" else small * 1.0e-4
        else:
            inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
            inp[0, :] = vec
            inp[1, :] = 0.0 if case == "rankdef" else small * 1.0e-4

    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_finite(res_u, res_s, res_v)
    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_v)


@pytest.mark.svd
@pytest.mark.parametrize("shape", utils.SVD_FALLBACK_SHAPES)
@pytest.mark.parametrize("some", [True, False])
def test_accuracy_svd_fallback_modes(shape, some):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=False)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=False)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=5e-4)


@pytest.mark.svd
def test_accuracy_svd_non_contiguous_empty_and_complex():
    inputs = [
        _make_input((3, 5), torch.float32).mT,
        torch.empty((0, 3), dtype=torch.float32, device=flag_gems.device),
        torch.empty((2, 3, 0), dtype=torch.float32, device=flag_gems.device),
        _make_input((3, 3), torch.complex64),
    ]

    for inp in inputs:
        ref_inp = utils.to_reference(inp, False)
        ref_u, ref_s, ref_v = torch.svd(ref_inp)
        with flag_gems.use_gems(include=["svd"]):
            res_u, res_s, res_v = torch.svd(inp)

        _assert_same_shape(res_u, ref_u)
        _assert_same_shape(res_s, ref_s)
        _assert_same_shape(res_v, ref_v)
        utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(3, 3), (2, 3, 3)])
def test_accuracy_svd_complex64_reconstruction(shape):
    inp = _make_input(shape, torch.complex64)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-2)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-2)


@pytest.mark.svd
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_svd_low_precision_float32_route(dtype):
    if dtype is torch.bfloat16 and not utils.bf16_is_supported:
        pytest.skip("bfloat16 is not supported on this device")

    inp = _make_input((5, 3), dtype)
    ref_inp = utils.to_reference(inp.to(torch.float32), False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    assert res_u.dtype == dtype
    assert res_s.dtype == dtype
    assert res_v.dtype == dtype
    atol = 7e-2 if dtype is torch.bfloat16 else 8e-3
    reconstructed = _reconstruct(
        res_u.to(torch.float32),
        res_s.to(torch.float32),
        res_v.to(torch.float32),
    )
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=atol)


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (4, 4)])
def test_accuracy_svd_some_false_reconstruction(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=False, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=False, compute_uv=True)

    _assert_same_shape(res_u, ref_u)
    _assert_same_shape(res_s, ref_s)
    _assert_same_shape(res_v, ref_v)
    utils.gems_assert_close(res_s, ref_s, res_s.dtype, atol=2e-3)
    reconstructed = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(reconstructed, ref_inp, reconstructed.dtype, atol=2e-3)
    _assert_orthonormal(res_u)
    _assert_orthonormal(res_v)
