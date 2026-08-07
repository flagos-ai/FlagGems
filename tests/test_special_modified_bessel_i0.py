import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_i0
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # upcast=True: 参考计算用 float,并按框架约定处理设备
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_i0(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_i0(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_modified_bessel_i0_out
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_i0(ref_inp)

    out_act = torch.empty_like(inp)
    with flag_gems.use_gems():
        act_out = torch.special.modified_bessel_i0(inp, out=out_act)

    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, ref_out, dtype)
