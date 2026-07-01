import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.alias_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alias_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.alias_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.alias_copy(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.alias_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alias_copy_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out_buf = torch.empty_like(ref_inp)
    act_out_buf = torch.empty_like(inp)
    ref_out = torch.ops.aten.alias_copy(ref_inp, out=ref_out_buf)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.alias_copy(inp, out=act_out_buf)
    utils.gems_assert_equal(act_out, ref_out)
