import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.add_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_relu(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # _add_relu computes relu(a + b) = max(0, a + b)
    # Since torch._add_relu is not available on CUDA, use relu(add(...)) as reference
    ref_out = torch.relu(inp1 + inp2)
    with flag_gems.use_gems():
        res_out = torch._add_relu(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
