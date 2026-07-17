import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.log2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log2(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log2(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log2_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.log2_()
    with flag_gems.use_gems():
        res_out = inp.log2_()

    utils.gems_assert_close(res_out, ref_out, dtype)
