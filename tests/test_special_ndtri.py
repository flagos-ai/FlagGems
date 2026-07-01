import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    SPECIAL_NDTRI_SHAPES = [(2, 3)]
else:
    SPECIAL_NDTRI_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.special_ndtri
@pytest.mark.parametrize("shape", SPECIAL_NDTRI_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_ndtri(shape, dtype):
    # input domain is (0, 1); clamp away from the exact 0/1 endpoints
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.998 + 0.001
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_ndtri(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_ndtri(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_ndtri(x)
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.special_ndtri_out
@pytest.mark.parametrize("shape", SPECIAL_NDTRI_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_ndtri_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 0.998 + 0.001
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_ndtri.out(ref_x.float(), out=out_ref)
        ref_out = out_ref.to(dtype)
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_ndtri.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_ndtri.out(x, out=out_act)
    assert act_out is out_act, "special_ndtri.out should return the out tensor"
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
