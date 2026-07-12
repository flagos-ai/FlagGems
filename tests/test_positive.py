import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.positive
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_positive(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = ref_inp

    with flag_gems.use_gems():
        res_out = flag_gems.positive(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

    assert (
        res_inp.data_ptr() == res_out.data_ptr()
    ), "positive must return direct reference to input tensor"
