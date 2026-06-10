import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.trunc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.trunc(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.trunc(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.trunc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trunc_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.trunc_()
    with flag_gems.use_gems():
        res_out = inp.trunc_()

    utils.gems_assert_equal(res_out, ref_out)
