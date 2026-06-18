import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_linalg_slogdet(n, dtype):
    # Build a well-conditioned random square matrix.
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
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_linalg_slogdet_batched(batch, n, dtype):
    shape = (batch, n, n)
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(a, True)

    ref_sign, ref_logabsdet = torch.linalg.slogdet(ref_inp)
    with flag_gems.use_gems():
        res_sign, res_logabsdet = torch.linalg.slogdet(a)

    utils.gems_assert_close(res_sign.to(ref_sign.dtype), ref_sign, ref_sign.dtype)
    utils.gems_assert_close(res_logabsdet, ref_logabsdet, dtype)
