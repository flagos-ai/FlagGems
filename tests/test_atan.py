import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.atan
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.atan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.atan(res_inp)
    ref_out = ref_out.to(res_out.dtype)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atan_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.atan_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.atan_(res_inp)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)
