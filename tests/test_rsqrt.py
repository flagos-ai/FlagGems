import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.rsqrt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rsqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.rsqrt_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
