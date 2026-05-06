import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.addcdiv
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcdiv(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = torch.addcdiv(ref_inp, ref_t1, ref_t2, value=v)
    with flag_gems.use_gems():
        res_out = torch.addcdiv(res_inp, t1, t2, value=v)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.addcdiv_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcdiv_out(dtype):
    shape = (64, 128)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device).clamp_min(1e-3)
    value = 0.5

    ref_inp = utils.to_reference(inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)
    ref_out = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    torch.ops.aten.addcdiv.out(ref_inp, ref_t1, ref_t2, value=value, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.addcdiv.out(inp, t1, t2, value=value, out=out)

    utils.gems_assert_close(out, ref_out, dtype)
