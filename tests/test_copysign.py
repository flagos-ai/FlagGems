import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.copysign
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign(shape, dtype):
    # Test copysign: magnitude of input, sign of other
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    ref_out = torch.copysign(ref_input, ref_other)

    with flag_gems.use_gems():
        res_out = torch.copysign(input, other)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.copysign_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_out(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(input)

    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    ref_out = torch.empty_like(ref_input)

    torch.copysign(ref_input, ref_other, out=ref_out)
    with flag_gems.use_gems():
        torch.copysign(input, other, out=out)

    utils.gems_assert_close(out, ref_out, dtype)
