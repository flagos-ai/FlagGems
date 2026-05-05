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
def test_logaddexp_broadcast(dtype):
    x = torch.randn((4, 1, 8), dtype=dtype, device=flag_gems.device)
    y = torch.randn((1, 6, 8), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_large_values(dtype):
    # exp(50) overflows fp16/bf16 — the shifted form must keep results finite.
    x = torch.tensor([50.0, -50.0, 30.0, -30.0], dtype=dtype, device=flag_gems.device)
    y = torch.tensor([49.0, -49.0, 35.0, -25.0], dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_edge_cases(dtype):
    x = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    y = torch.tensor(
        [0.0, 0.0, -1.0, 1.0, float("inf"), 1.0, 1.0],
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
def test_logaddexp_empty_tensor():
    x = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    y = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    ref_out = torch.logaddexp(x, y)
    with flag_gems.use_gems():
        res_out = torch.logaddexp(x, y)
    utils.gems_assert_equal(res_out, ref_out)


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
