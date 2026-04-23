import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.celu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_celu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.celu(ref_inp, alpha)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.celu(inp, alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.celu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_celu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    res_inp = inp.clone().to(flag_gems.device)
    inp_clone = inp.clone()
    ref_inp = utils.to_reference(inp_clone, True)
    torch.nn.functional.celu_(ref_inp, alpha)

    with flag_gems.use_gems():
        torch.nn.functional.celu_(res_inp, alpha)

    utils.gems_assert_close(res_inp, ref_inp, dtype)
