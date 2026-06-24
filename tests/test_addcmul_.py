import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.addcmul_
@pytest.mark.parametrize(
    "self_shape,t1_shape,t2_shape",
    [
        ((2, 3), (2, 3), (2, 3)),
        ((2, 3), (2, 1), (1, 3)),
        ((128, 256), (128, 256), (128, 1)),
        ((128, 256), (1, 256), (128, 256)),
        ((64, 128), (64, 1), (1, 128)),
        ((4, 8, 16), (1, 8, 1), (4, 1, 16)),
        ((512, 512), (512, 512), (512, 512)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("value", [1.0, 0.5, 2.0, -1.5])
def test_addcmul_(self_shape, t1_shape, t2_shape, dtype, value):
    inp = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(t1_shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(t2_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)
    ref_out = torch.ops.aten.addcmul_(ref_inp, ref_t1, ref_t2, value=value)

    inp1 = inp.clone()
    t1_copy = t1.clone()
    t2_copy = t2.clone()
    with flag_gems.use_gems():
        res_out = torch.ops.aten.addcmul_(inp1, t1_copy, t2_copy, value=value)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp, dtype)
    assert res_out is inp1
