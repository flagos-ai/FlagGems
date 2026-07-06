import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arctan
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arctan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arctan(inp)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.arctan_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arctan_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.arctan_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arctan_(inp)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)
