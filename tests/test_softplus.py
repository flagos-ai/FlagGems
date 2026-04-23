import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.softplus
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(inp, beta=beta, threshold=threshold)

    utils.gems_assert_close(res_out, ref_out, dtype)
