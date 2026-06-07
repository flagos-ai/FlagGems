import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.bucketize(ref_inp, boundaries)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize_right(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.bucketize(ref_inp, boundaries, right=True)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries, right=True)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bucketize
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_bucketize_int32(shape, dtype):
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=flag_gems.device)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.bucketize(ref_inp, boundaries, out_int32=True)

    with flag_gems.use_gems():
        res_out = torch.bucketize(inp, boundaries, out_int32=True)

    utils.gems_assert_equal(res_out, ref_out)
