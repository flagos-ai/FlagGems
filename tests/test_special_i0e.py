import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i0e(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_i0e(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_i0e(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_i0e(x)
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.special_i0e_out
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i0e_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_i0e.out(ref_x.float(), out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_i0e.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_i0e.out(x, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
