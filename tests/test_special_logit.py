import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_logit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logit_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_out(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        torch.special.logit(inp, eps=1e-6, out=out)
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.special_logit_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        flag_gems.special_logit_(inp, eps=1e-6)
    utils.gems_assert_close(inp, ref_out, dtype)
