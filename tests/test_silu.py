import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.silu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp, inplace=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_silu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.silu_backward(ref_grad, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.silu_backward(res_grad, res_inp)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
