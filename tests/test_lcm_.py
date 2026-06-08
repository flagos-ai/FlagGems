import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lcm
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_lcm(shape, dtype):
    # Avoid zeros in inputs as they can cause issues
    inp1 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.lcm(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.lcm(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.lcm_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_lcm_(shape, dtype):
    # Avoid zeros in inputs as they can cause issues
    inp1 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = torch.randint(low=1, high=100, size=shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.lcm_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.lcm_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)
