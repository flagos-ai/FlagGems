import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lerp_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)
    ref_weight = utils.to_reference(weight)

    ref_out = torch.lerp(ref_input, ref_end, weight=5.0)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.lerp(ref_input, ref_end, weight=ref_weight)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=weight)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)
    ref_weight = utils.to_reference(weight)

    ref_out = ref_input.clone().lerp_(ref_end, weight=5.0)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = ref_input.clone().lerp_(ref_end, weight=ref_weight)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=weight)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_scalar(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)

    ref_out = torch.lerp(ref_input, ref_end, weight=5.0)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=5.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.lerp(ref_input, ref_end, weight=-2.0)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=-2.0)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.lerp_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lerp_scalar_(shape, dtype):
    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)

    ref_input = utils.to_reference(input)
    ref_end = utils.to_reference(end)

    ref_out = ref_input.clone().lerp_(ref_end, weight=0.2)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=0.2)
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = ref_input.clone().lerp_(ref_end, weight=0.8)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=0.8)
    utils.gems_assert_close(res_out, ref_out, dtype)
