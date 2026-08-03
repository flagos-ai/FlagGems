import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.igamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_igamma_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)

    ref_out = ref_inp.igamma_(ref_other)
    with flag_gems.use_gems():
        res_out = inp.igamma_(other)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
