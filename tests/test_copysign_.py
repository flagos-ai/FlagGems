import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.copysign_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_(shape, dtype):
    # Test copysign_: in-place modification of first tensor
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = ref_input.copysign_(ref_other)

    with flag_gems.use_gems():
        res_out = input.copysign_(other)

    utils.gems_assert_close(res_out, ref_out, dtype)
