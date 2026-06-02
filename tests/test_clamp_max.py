import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp_max(ref_inp, max)
    with flag_gems.use_gems():
        res_out = torch.clamp_max(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.clamp_max_(max)
    with flag_gems.use_gems():
        res_out = inp.clamp_max_(max)

    utils.gems_assert_close(res_out, ref_out, dtype)
