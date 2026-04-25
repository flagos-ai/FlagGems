import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sigmoid
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sigmoid(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sigmoid_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sigmoid_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.sigmoid_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid_(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sigmoid_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sigmoid_backward(shape, dtype):
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_out)

    ref_out = utils.to_reference(res_out, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.sigmoid_backward(ref_grad, ref_out)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.sigmoid_backward(res_grad, res_out)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
