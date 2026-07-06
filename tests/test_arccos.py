import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arccos
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arccos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arccos(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arccos_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arccos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.arccos_()
    with flag_gems.use_gems():
        res_out = inp.arccos_()

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_inp, dtype, True)
