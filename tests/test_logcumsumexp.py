import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

LOGCUMSUMEXP_SHAPES = (
    [(2, 32)]
    if utils.QUICK_MODE
    else utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


@pytest.mark.logcumsumexp
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.logcumsumexp(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.logcumsumexp(inp, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.logcumsumexp_out
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logcumsumexp_out(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten.logcumsumexp.out(ref_inp, dim, out=ref_out_buf)

    res_out_buf = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.logcumsumexp.out(inp, dim, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])
