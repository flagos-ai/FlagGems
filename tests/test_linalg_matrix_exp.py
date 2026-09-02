import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

MATRIX_EXP_SHAPES = (
    [(2, 2), (16, 16), (128, 128)]
    if QUICK_MODE
    else [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ]
)
MATRIX_EXP_BATCH_SHAPES = (
    [(2, 3, 16, 16)]
    if QUICK_MODE
    else [
        (2, 4, 4),
        (3, 8, 8),
        (2, 3, 16, 16),
        (4, 2, 32, 32),
        (3, 64, 64),
        (4096, 4, 4),
        (1024, 8, 8),
        (1024, 16, 16),
        (128, 16, 16),
        (4, 32, 32),
        (512, 32, 32),
        (256, 64, 64),
        (32, 128, 128),
        (8, 256, 256),
    ]
)

MATRIX_EXP_DTYPES = [torch.float32] + (
    [torch.float64] if utils.fp64_is_supported else []
)


def _scaled_randn(shape, dtype):
    n = shape[-1]
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) / math.sqrt(n)


def _small_ops_matrix_exp(A):
    n = A.shape[-1]
    if n == 0:
        return A.clone()
    batch_shape = A.shape[:-2]
    B = math.prod(batch_shape) if batch_shape else 1
    A_flat = A.reshape(B, n, n)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    out = torch.empty_like(A_flat)
    for i in range(B):
        m = A_flat[i]
        norm = m.abs().sum(-2).max().item()
        s = math.ceil(math.log2(norm)) if norm > 1.0 else 0
        ms = m * (2.0**-s)
        r = eye.clone()
        term = eye.clone()
        for k in range(1, 31):
            term = (term @ ms) / k
            r = r + term
        for _ in range(s):
            r = r @ r
        out[i] = r
    return out.reshape(batch_shape + (n, n) if batch_shape else (n, n))


def _ref_matrix_exp(A):
    if A.device.type == "npu":
        return _small_ops_matrix_exp(A)
    prev = torch.get_num_threads()
    torch.set_num_threads(min(prev, 64))
    try:
        return torch.linalg.matrix_exp(A)
    finally:
        torch.set_num_threads(prev)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", MATRIX_EXP_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_random(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", MATRIX_EXP_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_random_batch(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


MATRIX_EXP_LARGE_NORM_CASES = [
    (dtype, scale)
    for dtype in MATRIX_EXP_DTYPES
    for scale in ([4.0] if dtype == torch.float32 else [4.0, 16.0, 64.0])
]


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (16, 16), (3, 32, 32), (64, 64)])
@pytest.mark.parametrize("dtype,scale", MATRIX_EXP_LARGE_NORM_CASES)
def test_linalg_matrix_exp_large_norm(shape, dtype, scale):
    n = shape[-1]
    A = _scaled_randn(shape, dtype) * scale

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (16, 16), (2, 3, 3)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_tiny_norm(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype) * 1e-8

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3), (2, 16, 16)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_nan(shape, dtype):
    A = _scaled_randn(shape, dtype)
    A[..., 0, 0] = float("nan")

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    assert torch.isnan(ref_out).all()
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (16, 16)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_batch_consistency(shape, dtype):
    A = _scaled_randn((3,) + tuple(shape), dtype)

    res_batch = flag_gems.linalg_matrix_exp(A)
    for i in range(3):
        res_single = flag_gems.linalg_matrix_exp(A[i])
        utils.gems_assert_equal(res_batch[i], res_single)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3), (2, 3, 16, 16)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_diagonal(shape, dtype):
    n = shape[-1]
    diag = torch.arange(1, n + 1, dtype=dtype, device=flag_gems.device) * 0.25
    diag[0] = -diag[0]
    A = torch.diag(diag)
    if len(shape) > 2:
        A = A.unsqueeze(0).expand(shape).contiguous()

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3), (16, 16)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_zero(shape, dtype):
    A = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("shape", [(1, 1), (4, 1, 1), (2, 3, 1, 1)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_one_by_one(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_empty(dtype):
    for shape in [(0, 0), (3, 0, 0), (0, 3, 3)]:
        A = torch.empty(shape, dtype=dtype, device=flag_gems.device)

        ref_A = utils.to_reference(A)
        ref_out = _ref_matrix_exp(ref_A)

        res_out = flag_gems.linalg_matrix_exp(A)

        assert res_out.shape == ref_out.shape
        assert res_out.dtype == dtype
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_matrix_exp
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_non_contiguous(dtype):
    n = 16
    base = _scaled_randn((2, n, n), dtype)
    A = base.transpose(-2, -1)
    assert not A.is_contiguous()

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    res_out = flag_gems.linalg_matrix_exp(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp
def test_linalg_matrix_exp_errors():
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.linalg_matrix_exp(torch.randn(3, 4, device=flag_gems.device))

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.linalg_matrix_exp(torch.randn(3, device=flag_gems.device))

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.linalg_matrix_exp(
            torch.randn(4, 4, dtype=torch.int32, device=flag_gems.device)
        )


@pytest.mark.linalg_matrix_exp_out
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3), (2, 3, 16, 16), (128, 128)])
@pytest.mark.parametrize("dtype", MATRIX_EXP_DTYPES)
def test_linalg_matrix_exp_out(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = _ref_matrix_exp(ref_A)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res = flag_gems.linalg_matrix_exp_out(A, out=out)

    assert res.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_matrix_exp_out
def test_linalg_matrix_exp_out_errors():
    A = torch.randn(4, 4, device=flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.linalg_matrix_exp_out(
            A, out=torch.empty((4, 4), dtype=torch.int32, device=flag_gems.device)
        )

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.linalg_matrix_exp_out(
            A, out=torch.empty((4, 5), dtype=A.dtype, device=flag_gems.device)
        )
