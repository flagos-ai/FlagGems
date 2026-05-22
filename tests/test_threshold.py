import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.threshold
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_threshold(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)
    threshold = 0
    value = 100

    ref_out = torch.nn.functional.threshold(ref_inp, threshold, value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.threshold(res_inp, threshold, value)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.threshold_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_threshold_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    threshold = 0

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.threshold_backward(ref_grad, ref_inp, threshold)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.threshold_backward(res_grad, res_inp, threshold)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
