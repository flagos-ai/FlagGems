import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

DET_SHAPES = (
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
DET_BATCH_SHAPES = (
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
DET_DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])


def _scaled_randn(shape, dtype):
    n = shape[-1]
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) / math.sqrt(n)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", DET_SHAPES)
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_random(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", DET_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_random_batch(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", [(4, 4), (16, 16), (2, 32, 32)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_positive_definite(shape, dtype):
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A = (B @ B.transpose(-2, -1) + n * eye) / (2 * n)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_negative_determinant(shape, dtype):
    n = shape[-1]
    A = torch.eye(n, dtype=dtype, device=flag_gems.device)
    A[[0, 1]] = A[[1, 0]]
    if len(shape) > 2:
        A = A.unsqueeze(0).expand(shape).contiguous()

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_diagonal(shape, dtype):
    n = shape[-1]
    diag = torch.arange(1, n + 1, dtype=dtype, device=flag_gems.device)
    diag[0] = -diag[0]
    A = torch.diag(diag)
    if len(shape) > 2:
        A = A.unsqueeze(0).expand(shape).contiguous()

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (2, 3, 3)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_singular(shape, dtype):
    A = _scaled_randn(shape, dtype)
    A[..., 0, :] = A[..., 1, :]

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_det
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_empty(dtype):
    for shape in [(0, 0), (3, 0, 0)]:
        A = torch.empty(shape, dtype=dtype, device=flag_gems.device)

        ref_A = utils.to_reference(A)
        ref_out = torch.linalg.det(ref_A)

        with flag_gems.use_gems():
            res_out = torch.linalg.det(A)

        utils.gems_assert_close(res_out, ref_out, dtype)

    A = torch.empty((0, 3, 3), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)
    assert res_out.shape == (0,)
    assert res_out.dtype == dtype


@pytest.mark.linalg_det
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_non_contiguous(dtype):
    n = 16
    base = _scaled_randn((n, n), dtype)
    A = base.transpose(-2, -1)
    assert not A.is_contiguous()

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.det(A)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det
def test_linalg_det_errors():
    with flag_gems.use_gems():
        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.det(torch.randn(3, 4, device=flag_gems.device))

        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.det(torch.randn(3, device=flag_gems.device))


@pytest.mark.linalg_det_out
@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3), (2, 3, 16, 16), (128, 128)])
@pytest.mark.parametrize("dtype", DET_DTYPES)
def test_linalg_det_out(shape, dtype):
    n = shape[-1]
    A = _scaled_randn(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.det(ref_A)

    out = torch.empty(shape[:-2], dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res = torch.linalg.det(A, out=out)

    assert res.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=n)


@pytest.mark.linalg_det_out
def test_linalg_det_out_errors():
    A = torch.randn(4, 4, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.det(
                A, out=torch.empty((), dtype=torch.int32, device=flag_gems.device)
            )

        with pytest.raises((RuntimeError, ValueError)):
            torch.linalg.det(
                A, out=torch.empty((4,), dtype=A.dtype, device=flag_gems.device)
            )
