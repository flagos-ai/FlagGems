import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.adaptive_attention_span
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX backend path for adaptive_attention_span is not stable in CI. See issue #3984.",
)
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_attention_span(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = flag_gems.adaptive_attention_span(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)
