import pytest
import torch

from flag_gems.testing import utils


@pytest.mark.negative
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_negative(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda") if dtype in utils.FLOAT_DTYPES else \
          torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    ref_inp = inp.clone()

    res = torch.ops.aten.negative(inp)
    ref = torch.negative(ref_inp)

    utils.gems_assert_equal(res, ref)


@pytest.mark.negative
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_negative_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda") if dtype in utils.FLOAT_DTYPES else \
          torch.randint(-100, 100, shape, dtype=dtype, device="cuda")
    ref_inp = inp.clone()

    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.negative.out(inp, out=out)
    torch.negative(ref_inp, out=ref_out)

    utils.gems_assert_equal(out, ref_out)
