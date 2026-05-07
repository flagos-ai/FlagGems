import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.gelu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(res_inp, approximate=approximate)

    atol = 1e-4
    if flag_gems.vendor_name == "aipu" and dtype == torch.float16:
        atol = 1e-3
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.gelu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_inp)

    ref_inp = utils.to_reference(res_inp, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.ops.aten.gelu_backward(
        ref_out, ref_inp, approximate=approximate
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.gelu_backward(
            res_out, res_inp, approximate=approximate
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.gelu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp.clone(), True)

    ref_out = torch.ops.aten.gelu_.default(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.gelu_.default(res_inp, approximate=approximate)

    utils.gems_assert_close(res_out, ref_out, dtype)
