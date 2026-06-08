import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES


@pytest.mark.atan
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
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
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_atan_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.atan_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.atan_(res_inp)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype)
