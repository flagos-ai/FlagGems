import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


FAST_SHAPES = [(2, 2), (8, 2), (2, 8), (16, 8), (8, 16), (64, 32), (32, 64)]
FALLBACK_SHAPES = [(5, 3), (3, 5), (2, 4, 4)]


def _make_input(shape, dtype):
    if dtype.is_complex:
        real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
        return (real + 1j * imag).to(dtype)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _reconstruct(u, s, v):
    k = s.shape[-1]
    return u[..., :, :k] @ torch.diag_embed(s).to(u.dtype) @ v[..., :, :k].mH


def _assert_close(actual, expected, atol=5e-4, rtol=5e-4):
    torch.testing.assert_close(
        utils.to_cpu(actual, expected),
        expected,
        atol=atol,
        rtol=rtol,
        check_dtype=False,
    )


@pytest.mark.svd
@pytest.mark.parametrize("shape", FAST_SHAPES)
def test_accuracy_svd_fast_float32(shape):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=True, compute_uv=True)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=True, compute_uv=True)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_s, ref_s, atol=2e-3, rtol=2e-3)
    _assert_close(_reconstruct(res_u, res_s, res_v), ref_inp, atol=2e-3, rtol=2e-3)


@pytest.mark.svd
@pytest.mark.parametrize("shape", FALLBACK_SHAPES)
@pytest.mark.parametrize("some", [True, False])
def test_accuracy_svd_fallback_modes(shape, some):
    inp = _make_input(shape, torch.float32)
    ref_inp = utils.to_reference(inp, False)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=False)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=False)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    _assert_close(res_s, ref_s)


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

        assert res_u.shape == ref_u.shape
        assert res_s.shape == ref_s.shape
        assert res_v.shape == ref_v.shape
        _assert_close(res_s, ref_s, atol=2e-3, rtol=2e-3)
