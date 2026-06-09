import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2.0 - 1.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.arcsin_()
    with flag_gems.use_gems():
        inp.arcsin_()

    utils.gems_assert_close(inp, ref_inp, dtype)
    utils.gems_assert_close(inp, ref_out, dtype)
