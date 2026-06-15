import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.soft_margin_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss(shape, dtype, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (torch.randint(0, 2, shape, device=flag_gems.device).to(dtype) * 2) - 1

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.soft_margin_loss(ref_inp, ref_target, reduction)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.soft_margin_loss(inp, target, reduction)

    utils.gems_assert_close(res_out, ref_out, dtype)
