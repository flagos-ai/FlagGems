import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.block_diag
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_block_diag(dtype):
    # Test with multiple blocks of different sizes
    a = torch.randn((2, 3), dtype=dtype, device=flag_gems.device)
    b = torch.randn((4, 2), dtype=dtype, device=flag_gems.device)
    c = torch.randn((1, 5), dtype=dtype, device=flag_gems.device)

    ref_a = utils.to_reference(a)
    ref_b = utils.to_reference(b)
    ref_c = utils.to_reference(c)

    ref_out = torch.block_diag(ref_a, ref_b, ref_c)
    with flag_gems.use_gems():
        res_out = torch.block_diag(a, b, c)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.block_diag
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_block_diag_square(dtype):
    # Test with square matrices of varying sizes
    blocks = [
        torch.randn((sz, sz), dtype=dtype, device=flag_gems.device)
        for sz in [16, 64, 128]
    ]
    refs = [utils.to_reference(b) for b in blocks]

    ref_out = torch.block_diag(*refs)
    with flag_gems.use_gems():
        res_out = torch.block_diag(*blocks)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.block_diag
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_block_diag_1d_and_scalar(dtype):
    # Test with 1D tensors and scalars mixed with 2D
    a = torch.randn((3,), dtype=dtype, device=flag_gems.device)
    b = torch.tensor(5.0, dtype=dtype, device=flag_gems.device)
    c = torch.randn((2, 4), dtype=dtype, device=flag_gems.device)

    ref_a = utils.to_reference(a)
    ref_b = utils.to_reference(b)
    ref_c = utils.to_reference(c)

    ref_out = torch.block_diag(ref_a, ref_b, ref_c)
    with flag_gems.use_gems():
        res_out = torch.block_diag(a, b, c)

    utils.gems_assert_close(res_out, ref_out, dtype)
