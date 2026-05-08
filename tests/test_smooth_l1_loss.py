import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test with FLOAT_DTYPES: float16, float32, bfloat16 (skip float64 if not supported)
FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize(
    "shape",
    [
        # 1D
        (8,),
        (64,),
        # 2D
        (32, 32),
        (128, 256),
        # 3D
        (4, 16, 32),
        (64, 64, 64),
        # 4D
        (2, 3, 4, 5),
        (8, 16, 32, 64),
        # 5D
        (2, 3, 4, 5, 6),
        (4, 8, 16, 32, 64),
    ],
)
def test_smooth_l1_loss(shape, dtype, reduction):
    """Test smooth_l1_loss across various shapes, dtypes, and reductions."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.1, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("shape", [(32, 32), (64, 128)])
def test_smooth_l1_loss_beta(shape, dtype, reduction, beta):
    """Test smooth_l1_loss with various beta values."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, beta)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0)])
def test_smooth_l1_loss_empty(shape, dtype, reduction):
    """Test smooth_l1_loss with empty tensors."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_smooth_l1_loss_identical(shape, dtype, reduction):
    """Test smooth_l1_loss when input equals target (loss should be zero)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = inp.clone()

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_smooth_l1_loss_large_diff(shape, dtype, reduction):
    """Test smooth_l1_loss with large differences (hits linear regime)."""
    inp = torch.full(shape, 100.0, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_small_diff(dtype, reduction):
    """Test smooth_l1_loss with small differences (hits quadratic regime)."""
    shape = (64, 64)
    inp = torch.full(shape, 0.01, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
