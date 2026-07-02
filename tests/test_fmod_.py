import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.fmod_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# fmod only supports float32 due to integer division semantics
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = utils.to_reference(inp.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.fmod_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp.fmod_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=2.0)
    utils.gems_assert_close(inp, ref_out, dtype, atol=2.0)

    ref_inp1 = utils.to_reference(inp.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1.fmod_(ref_d)
        with flag_gems.use_gems():
            res_out = inp.clone().fmod_(d)
        utils.gems_assert_close(res_out, ref_out, dtype, atol=2.0)
