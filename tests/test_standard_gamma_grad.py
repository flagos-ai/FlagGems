import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_grad(shape, dtype):
    # Generate positive values for alpha (self_grad) and output
    # since gamma distribution parameters must be positive
    res_self_grad = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    res_output = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems._standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_grad_near_mode(shape, dtype):
    # Large alpha (> 8) with output near the distribution mode exercises the
    # well-conditioned near-mode sub-case of the Rice saddle-point branch.
    res_self_grad = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * 40.0 + 10.0
    )
    res_output = res_self_grad * (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.2 + 0.9
    )

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems._standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_standard_gamma_grad_small_output(shape, dtype):
    # Small output (< 0.8) exercises the Taylor-series branch.
    res_self_grad = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    res_output = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.7 + 0.01

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems._standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)
