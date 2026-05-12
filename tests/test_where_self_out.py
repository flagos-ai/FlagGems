import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.where_self
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp1 > 0, ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp1 > 0, inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self_scalar(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self_scalar_other(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, ref_inp2, inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp2, inp1)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(out)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)
    ref_cond = utils.to_reference(cond)

    ref_out = torch.where(ref_cond, ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.where(cond, inp1, inp2, out=out)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.where_self_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_where_self_out_cross_device(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

    import itertools

    shapes = (shape, None)
    for a_shape, b_shape, c_shape in itertools.product(shapes, shapes, shapes):
        a = inp1 if a_shape else torch.tensor(0)
        b = inp2 if b_shape else torch.tensor(1)
        c = cond if c_shape else torch.tensor(True)

        ref_a = utils.to_reference(a)
        ref_b = utils.to_reference(b)
        ref_c = utils.to_reference(c)

        ref_out = torch.where(ref_c, ref_a, ref_b)
        with flag_gems.use_gems():
            res_out = torch.where(c, a, b)

        utils.gems_assert_equal(res_out, ref_out)
