import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.fmax(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.fmax(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_with_nan(shape, dtype):
    # Inject NaNs into both tensors to exercise the asymmetric NaN handling:
    # if exactly one input is NaN, fmax returns the other; both NaN -> NaN.
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if x.numel() >= 4:
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_flat[0] = float("nan")
        y_flat[1] = float("nan")
        x_flat[2] = float("nan")
        y_flat[2] = float("nan")

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.fmax(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.fmax(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_broadcast(shape, dtype):
    if len(shape) < 2:
        pytest.skip("broadcast variant needs >=2D shapes")
    x_shape = list(shape)
    y_shape = list(shape)
    y_shape[0] = 1

    x = torch.randn(x_shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.fmax(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.fmax(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmax
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fmax_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.fmax.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.fmax.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
