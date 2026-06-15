import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.selu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.selu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.selu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.selu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.selu_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.selu_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
