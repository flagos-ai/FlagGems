import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_special_values(dtype):
    x = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [0.0, 1.0, float("inf"), float("-inf"), float("nan"), 0.0, -1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_empty(dtype):
    for shape in [(0,), (4, 0), (2, 0, 3)]:
        x = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        y = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_x = utils.to_reference(x, True)
        ref_y = utils.to_reference(y, True)
        ref_out = torch.logaddexp(ref_x, ref_y)
        with flag_gems.use_gems():
            res_out = torch.logaddexp(x, y)
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_noncontiguous(dtype):
    base_x = torch.randn((33, 17), dtype=dtype, device=flag_gems.device)
    base_y = torch.randn((33, 17), dtype=dtype, device=flag_gems.device)
    x = base_x.transpose(-1, -2)
    y = base_y.transpose(-1, -2)
    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_broadcast(dtype):
    a = torch.rand((4, 1), dtype=dtype, device=flag_gems.device)
    b = torch.rand((1, 4), dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, True)
    ref_b = utils.to_reference(b, True)
    ref_out = torch.logaddexp(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(a, b)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logaddexp.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
