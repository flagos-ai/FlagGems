import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cast
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cast(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Cast is identity function - preserves the tensor values
    ref_out = ref_inp
    with flag_gems.use_gems():
        res_out = flag_gems.cast(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cast_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cast_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Cast_ is in-place identity function - preserves the tensor values
    with flag_gems.use_gems():
        flag_gems.cast_(inp)

    utils.gems_assert_close(inp, ref_inp, dtype)
