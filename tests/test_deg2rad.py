import pytest
import torch

from flag_gems.testing import utils


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = inp.clone()

    res = torch.ops.aten.deg2rad(inp)
    ref = torch.deg2rad(ref_inp)

    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = inp.clone()

    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.deg2rad.out(inp, out=out)
    torch.deg2rad(ref_inp, out=ref_out)

    utils.gems_assert_close(out, ref_out, dtype)
