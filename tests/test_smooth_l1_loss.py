import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLOAT_DTYPES = utils.FLOAT_DTYPES


# ===========================================================================
# smooth_l1_loss
# ===========================================================================
@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("shape", [(64,), (32, 32), (128, 256)])
def test_accuracy_smooth_l1_loss(shape, dtype, reduction):
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
@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("shape", [(32, 32)])
def test_accuracy_smooth_l1_loss_beta(shape, dtype, reduction, beta):
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
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_accuracy_smooth_l1_loss_empty(shape, dtype, reduction):
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
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_identical(shape, dtype, reduction):
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
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_large_diff(shape, dtype, reduction):
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
@pytest.mark.parametrize("reduction", [1, 2])
def test_accuracy_smooth_l1_loss_small_diff(dtype, reduction):
    shape = (64, 64)
    inp = torch.full(shape, 0.01, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# ===========================================================================
# smooth_l1_loss_out
# ===========================================================================
def _smooth_l1_loss_out_test(dtype, reduction, shape, beta=1.0):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    out_shape = shape if reduction == 0 else ()
    ref_out = torch.empty(out_shape, dtype=dtype, device=ref_inp.device)
    ref_out = ref_out.resize_(0)
    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, reduction, beta, out=ref_out)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss.out(inp, target, reduction, beta, out=out)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(64,), (32, 32), (128, 256)])
def test_accuracy_smooth_l1_loss_out(shape, dtype, reduction):
    _smooth_l1_loss_out_test(dtype, reduction, shape)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("shape", [(32, 32)])
def test_accuracy_smooth_l1_loss_out_beta(shape, dtype, reduction, beta):
    _smooth_l1_loss_out_test(dtype, reduction, shape, beta)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(0,), (3, 0)])
def test_accuracy_smooth_l1_loss_out_empty(shape, dtype, reduction):
    _smooth_l1_loss_out_test(dtype, reduction, shape)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_out_identical(shape, dtype, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = inp.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out.resize_(0)
    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, reduction, 1.0, out=ref_out)
    out = torch.empty((), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss.out(inp, target, reduction, 1.0, out=out)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("shape", [(64, 64)])
def test_accuracy_smooth_l1_loss_out_large_diff(shape, dtype, reduction):
    inp = torch.full(shape, 100.0, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out.resize_(0)
    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, reduction, 1.0, out=ref_out)
    out = torch.empty((), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss.out(inp, target, reduction, 1.0, out=out)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [1, 2])
def test_accuracy_smooth_l1_loss_out_small_diff(dtype, reduction):
    shape = (64, 64)
    inp = torch.full(shape, 0.01, dtype=dtype, device=flag_gems.device)
    target = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_out = ref_out.resize_(0)
    torch.ops.aten.smooth_l1_loss.out(ref_inp, ref_target, reduction, 1.0, out=ref_out)
    out = torch.empty((), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss.out(inp, target, reduction, 1.0, out=out)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
