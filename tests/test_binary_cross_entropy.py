import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_binary_cross_entropy(shape, dtype, reduction):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.binary_cross_entropy(ref_inp, ref_target, None, reduction)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.binary_cross_entropy(inp, target, None, reduction)
    reduce_dim = shape[-1] if reduction == 0 else 1
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_binary_cross_entropy_weight(shape, dtype, reduction):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight)
    ref_out = torch.ops.aten.binary_cross_entropy(
        ref_inp, ref_target, ref_weight, reduction
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.binary_cross_entropy(inp, target, weight, reduction)
    reduce_dim = shape[-1] if reduction == 0 else 1
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
