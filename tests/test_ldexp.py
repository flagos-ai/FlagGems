import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Keep the exponent in a moderate range so that the result stays
    # representable in fp16 / bf16 (their max ~6.5e4 / 3.4e38 respectively).
    exp = torch.randint(-6, 6, shape, device=flag_gems.device).to(dtype)

    ref_x = utils.to_reference(x, True)
    ref_exp = utils.to_reference(exp, True)
    ref_out = torch.ldexp(ref_x, ref_exp)

    with flag_gems.use_gems():
        res_out = torch.ldexp(x, exp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_zero_x(shape, dtype):
    # ldexp(0, anything finite) must be 0; the exponent never matters because
    # 0 * 2**k == 0.
    x = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    exp = torch.randint(-10, 10, shape, device=flag_gems.device).to(dtype)

    ref_x = utils.to_reference(x, True)
    ref_exp = utils.to_reference(exp, True)
    ref_out = torch.ldexp(ref_x, ref_exp)

    with flag_gems.use_gems():
        res_out = torch.ldexp(x, exp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_broadcast(shape, dtype):
    if len(shape) < 2:
        pytest.skip("broadcast variant needs >=2D shapes")
    x_shape = list(shape)
    exp_shape = list(shape)
    exp_shape[0] = 1

    x = torch.randn(x_shape, dtype=dtype, device=flag_gems.device)
    exp = torch.randint(-4, 4, exp_shape, device=flag_gems.device).to(dtype)

    ref_x = utils.to_reference(x, True)
    ref_exp = utils.to_reference(exp, True)
    ref_out = torch.ldexp(ref_x, ref_exp)

    with flag_gems.use_gems():
        res_out = torch.ldexp(x, exp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    exp = torch.randint(-4, 4, shape, device=flag_gems.device).to(dtype)

    ref_x = utils.to_reference(x, True)
    ref_exp = utils.to_reference(exp, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.ldexp.out(ref_x, ref_exp, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.ldexp.out(x, exp, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
