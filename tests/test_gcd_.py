import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.gcd_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_gcd_(shape, dtype):
    # GCD is only defined for integer types
    inp1 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.gcd_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.gcd_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
