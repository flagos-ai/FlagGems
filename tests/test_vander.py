import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.vander
@pytest.mark.parametrize("M", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("N", [None, 1, 3, 8, 32])
@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_vander(M, N, increasing, dtype):
    res_inp = torch.randn(M, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.vander(ref_inp, N=N, increasing=increasing)
    res_out = flag_gems.vander(res_inp, N=N, increasing=increasing)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.vander
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_vander_int(dtype):
    """Test vander with integer dtypes"""
    M = 8
    res_inp = torch.randint(0, 10, (M,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.vander(ref_inp, N=5)
    res_out = flag_gems.vander(res_inp, N=5)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.vander
def test_vander_edge_cases():
    """Test edge cases: empty tensor, single element"""
    # Empty tensor
    res_inp = torch.tensor([], dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.vander(ref_inp)
    res_out = flag_gems.vander(res_inp)
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    # Single element
    res_inp = torch.tensor([2.0], dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.vander(ref_inp, N=3)
    res_out = flag_gems.vander(res_inp, N=3)
    utils.gems_assert_close(res_out, ref_out, torch.float32)
