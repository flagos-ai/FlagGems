import pytest
import torch

from flag_gems.testing import utils


@pytest.mark.fix
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fix(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda") * 10  # 使用较大范围测试截断
    ref_inp = inp.clone()

    res = torch.ops.aten.fix(inp)
    ref = torch.fix(ref_inp)

    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.fix
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fix_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda") * 10
    ref_inp = inp.clone()

    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten.fix.out(inp, out=out)
    torch.fix(ref_inp, out=ref_out)

    utils.gems_assert_close(out, ref_out, dtype)
