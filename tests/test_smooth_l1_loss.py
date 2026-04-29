import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils

SMOOTH_L1_SHAPES = [
    (64,),
    (32, 32),
    (8, 8, 8),
    (16, 128, 64),
]

REDUCTION_MODES = ["none", "mean", "sum"]
BETA_VALUES = [0.5, 1.0, 2.0]


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", SMOOTH_L1_SHAPES)
@pytest.mark.parametrize("reduction", REDUCTION_MODES)
@pytest.mark.parametrize("beta", BETA_VALUES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss(shape, reduction, beta, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = F.smooth_l1_loss(ref_inp, ref_target, reduction=reduction, beta=beta)

    with flag_gems.use_gems():
        res_out = F.smooth_l1_loss(inp, target, reduction=reduction, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_zero_diff(dtype):
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = F.smooth_l1_loss(ref_inp, ref_inp, reduction="mean")

    with flag_gems.use_gems():
        res_out = F.smooth_l1_loss(inp, inp, reduction="mean")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_large(dtype):
    inp = torch.randn((256, 256), dtype=dtype, device=flag_gems.device)
    target = torch.randn((256, 256), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = F.smooth_l1_loss(ref_inp, ref_target, reduction="mean")

    with flag_gems.use_gems():
        res_out = F.smooth_l1_loss(inp, target, reduction="mean")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_noncontiguous(dtype):
    base = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)
    target = torch.randn_like(inp)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = F.smooth_l1_loss(ref_inp, ref_target, reduction="mean")

    with flag_gems.use_gems():
        res_out = F.smooth_l1_loss(inp, target, reduction="mean")

    utils.gems_assert_close(res_out, ref_out, dtype)
