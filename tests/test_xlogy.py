import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy(shape, dtype):
    # Positive y dominates here (log requires y > 0 for finite results) but we
    # also let randn flip negatives to cover NaN propagation.
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_zero_x(shape, dtype):
    # xlogy(0, y) must be exactly 0 regardless of y -- including y == 0,
    # y < 0, and y == NaN. This is the defining special case of the op.
    x = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    y_raw = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Inject zeros, negatives, and NaNs into y to exercise every branch.
    y = y_raw.clone()
    if y.numel() >= 4:
        flat = y.flatten()
        flat[0] = 0.0
        flat[1] = -1.0
        flat[2] = float("nan")
        flat[3] = float("inf")

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_broadcast(shape, dtype):
    # Broadcasting along a single dim verifies pointwise_dynamic handles the
    # usual NumPy-style shape expansion.
    if len(shape) < 2:
        pytest.skip("broadcast variant needs >=2D shapes")
    x_shape = list(shape)
    y_shape = list(shape)
    y_shape[0] = 1

    x = torch.rand(x_shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.1
    y = torch.rand(y_shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.xlogy(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.xlogy(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.xlogy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_xlogy_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 4.0 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.xlogy.OutTensor(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.xlogy.OutTensor(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
