import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.binary_cross_entropy_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_binary_cross_entropy_backward(shape, dtype, reduction):
    # Create inputs: self is probability (in (0,1)), target is binary labels
    # Use rand to generate values in (0, 1) - required for BCE backward
    self = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    # Avoid exact 0 or 1 values to prevent division by zero
    self = torch.clamp(self, 1e-4, 1 - 1e-4)
    target = torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device).to(
        dtype=dtype
    )
    grad_output = torch.ones_like(self)

    ref_self = utils.to_reference(self, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.binary_cross_entropy_backward(
        torch.ones_like(ref_self),
        ref_self,
        ref_target,
        None,
        reduction,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.binary_cross_entropy_backward(
            grad_output, self, target, None, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.binary_cross_entropy_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_binary_cross_entropy_backward_weighted(shape, dtype, reduction):
    # Test with weight tensor
    # Use rand to generate values in (0, 1) - required for BCE backward
    self = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    # Avoid exact 0 or 1 values to prevent division by zero
    self = torch.clamp(self, 1e-4, 1 - 1e-4)
    target = torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device).to(
        dtype=dtype
    )
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.ones_like(self)

    ref_self = utils.to_reference(self, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.ops.aten.binary_cross_entropy_backward(
        torch.ones_like(ref_self),
        ref_self,
        ref_target,
        ref_weight,
        reduction,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.binary_cross_entropy_backward(
            grad_output, self, target, weight, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
