import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_standard_gamma_grad(shape, dtype):
    # Generate positive values for alpha (self_grad) and output
    # since gamma distribution parameters must be positive
    res_self_grad = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    res_output = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems.standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
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
    res_out = flag_gems.standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_standard_gamma_grad_away_from_mode(shape, dtype):
    # Large alpha (> 8) with output away from the mode (below 0.9 * alpha and
    # above 1.1 * alpha) exercises the away-from-mode sub-case of the Rice
    # saddle-point branch, which the near-mode test does not reach.
    res_self_grad = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * 40.0 + 10.0
    )
    # Scale factor in [0.5, 0.85) U [1.15, 1.5): strictly outside [0.9, 1.1].
    scale = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.35
    below = 0.5 + scale
    above = 1.15 + scale
    pick_above = torch.rand(shape, device=flag_gems.device) < 0.5
    factor = torch.where(pick_above, above, below)
    res_output = res_self_grad * factor

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems.standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_standard_gamma_grad_small_output(shape, dtype):
    # Small output (< 0.8) exercises the Taylor-series branch.
    res_self_grad = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.5
    res_output = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.7 + 0.01

    ref_self_grad = utils.to_reference(res_self_grad, True)
    ref_output = utils.to_reference(res_output, True)

    ref_out = torch._standard_gamma_grad(ref_self_grad, ref_output)
    res_out = flag_gems.standard_gamma_grad(res_self_grad, res_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad_mismatched_dtype_raises():
    # Native _standard_gamma_grad requires matching floating dtypes and does not
    # promote mixed dtypes; the FlagGems op must reject the same inputs.
    self_grad = torch.rand((8,), dtype=torch.float32, device=flag_gems.device) + 0.5
    output = torch.rand((8,), dtype=torch.float64, device=flag_gems.device) + 0.5

    with pytest.raises(RuntimeError):
        torch._standard_gamma_grad(self_grad, output)
    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma_grad(self_grad, output)


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad_integer_dtype_raises():
    # Integer inputs are unsupported by native _standard_gamma_grad; the FlagGems
    # op must reject them rather than promoting them to a floating dtype.
    self_grad = torch.ones((8,), dtype=torch.int64, device=flag_gems.device)
    output = torch.ones((8,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch._standard_gamma_grad(self_grad, output)
    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma_grad(self_grad, output)
