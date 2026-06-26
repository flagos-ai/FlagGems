import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    I0_SHAPES = [(1024, 1024)]
else:
    I0_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)]


@pytest.mark.i0
@pytest.mark.parametrize("shape", I0_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_i0(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.i0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.i0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.i0_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_i0_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.i0_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.i0_(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.i0_out
@pytest.mark.parametrize("shape", I0_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_i0_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_inp, dtype=torch.float32)
        ref_out = torch.ops.aten.i0.out(ref_inp.float(), out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_inp)
        ref_out = torch.ops.aten.i0.out(ref_inp, out=out_ref)
    out_act = torch.empty_like(inp)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.i0.out(inp, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
