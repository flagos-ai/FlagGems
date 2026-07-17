import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.inplace
@pytest.mark.gt_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_gt_scalar_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    inp2 = 0

    ref_out = ref_inp1.gt_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.gt_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
