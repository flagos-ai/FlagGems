import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.binary_cross_entropy_with_logits
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_with_logits(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.binary_cross_entropy_with_logits(ref_inp, ref_target, reduction=0)
    with flag_gems.use_gems():
        res_out = torch.binary_cross_entropy_with_logits(inp, target, reduction=0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.binary_cross_entropy_with_logits
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_with_logits_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.binary_cross_entropy_with_logits(ref_inp, ref_target, reduction=1)
    with flag_gems.use_gems():
        res_out = torch.binary_cross_entropy_with_logits(inp, target, reduction=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.binary_cross_entropy_with_logits
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_with_logits_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.binary_cross_entropy_with_logits(ref_inp, ref_target, reduction=2)
    with flag_gems.use_gems():
        res_out = torch.binary_cross_entropy_with_logits(inp, target, reduction=2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.binary_cross_entropy_with_logits
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_with_logits_weight(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.binary_cross_entropy_with_logits(
        ref_inp, ref_target, weight=ref_weight, reduction=0
    )
    with flag_gems.use_gems():
        res_out = torch.binary_cross_entropy_with_logits(
            inp, target, weight=weight, reduction=0
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.binary_cross_entropy_with_logits
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_with_logits_pos_weight(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    pos_weight = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_pos_weight = utils.to_reference(pos_weight, True)

    ref_out = torch.binary_cross_entropy_with_logits(
        ref_inp, ref_target, pos_weight=ref_pos_weight, reduction=0
    )
    with flag_gems.use_gems():
        res_out = torch.binary_cross_entropy_with_logits(
            inp, target, pos_weight=pos_weight, reduction=0
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
