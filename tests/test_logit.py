import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.logit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.logit(inp, eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logit_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit_(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.logit_(eps=1e-6)
    with flag_gems.use_gems():
        res_out = inp.logit_(eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)
