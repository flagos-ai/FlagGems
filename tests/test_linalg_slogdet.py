import pytest
import torch

import flag_gems
from flag_gems.ops.linalg_slogdet import slogdet

from . import accuracy_utils as utils

# slogdet kernel only supports float32 and float64
SLOGDET_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet(n, dtype):
    shape = (n, n)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_batched(batch, n, dtype):
    shape = (batch, n, n)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_negative_det(dtype):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_singular(dtype):
    a = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_linalg_slogdet_row_swap(dtype):
    a = torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_slogdet_internal(dtype):
    n = 3
    a = torch.randn((n, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    sign, logabsdet, LU, pivots = slogdet(a)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    utils.gems_assert_close(sign.to(dtype), ref_sign.to(dtype), dtype)
    utils.gems_assert_close(logabsdet.to(dtype), ref_logabsdet.to(dtype), dtype)

    assert LU.shape == a.shape
    assert LU.dtype == a.dtype
    assert pivots.shape == (n,)
    assert pivots.dtype == torch.int32


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("dtype", SLOGDET_DTYPES)
def test_slogdet_batched_pivots(dtype):
    batch, n = 2, 3
    a = torch.randn((batch, n, n), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    sign, logabsdet, LU, pivots = slogdet(a)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    utils.gems_assert_close(sign.to(dtype), ref_sign.to(dtype), dtype)
    utils.gems_assert_close(logabsdet.to(dtype), ref_logabsdet.to(dtype), dtype)

    assert LU.shape == a.shape
    assert pivots.shape == (batch, n)
    assert pivots.dtype == torch.int32
