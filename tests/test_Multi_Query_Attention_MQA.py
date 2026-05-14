import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.Multi_Query_Attention_MQA
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_Multi_Query_Attention_MQA(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.Multi_Query_Attention_MQA(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.Multi_Query_Attention_MQA(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
