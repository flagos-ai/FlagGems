import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark._dyn_quant_pack_4bit_weight
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__dyn_quant_pack_4bit_weight(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch._dyn_quant_pack_4bit_weight(ref_inp)
    with flag_gems.use_gems():
        res_out = torch._dyn_quant_pack_4bit_weight(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
