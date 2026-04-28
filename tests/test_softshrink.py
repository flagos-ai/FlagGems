import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.softshrink
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("lambd", [0.5, 1.0, 0.0])
def test_softshrink(shape, dtype, lambd):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softshrink(ref_inp, lambd=lambd)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.softshrink(inp, lambd)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softshrink_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softshrink_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.softshrink.out(ref_inp, 0.5, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.softshrink.out(inp, 0.5, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)
