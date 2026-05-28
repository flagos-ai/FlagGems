import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.where_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self_backward(shape, dtype):
    """Test where_self_backward with various shapes."""
    condition = torch.randn(shape, dtype=dtype, device=flag_gems.device) > 0
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_cond = utils.to_reference(condition)
    ref_self = utils.to_reference(self)
    ref_other = utils.to_reference(other)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.where.Self_backward(ref_grad, ref_cond, ref_self, ref_other)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.where.Self_backward(grad_output, condition, self, other)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.where_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_other_backward(shape, dtype):
    """Test where_other_backward with various shapes."""
    condition = torch.randn(shape, dtype=dtype, device=flag_gems.device) > 0
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_cond = utils.to_reference(condition)
    ref_self = utils.to_reference(self)
    ref_other = utils.to_reference(other)
    ref_grad = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.where.Other_backward(ref_grad, ref_cond, ref_self, ref_other)

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.where.Other_backward(grad_output, condition, self, other)

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
