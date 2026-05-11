import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.margin_ranking_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_margin_ranking_loss(shape, dtype, margin, reduction):
    input1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    target = (
        torch.randint(0, 2, shape, device=flag_gems.device, dtype=torch.int8) * 2 - 1
    ).to(dtype)

    ref_input1 = utils.to_reference(input1)
    ref_input2 = utils.to_reference(input2)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.margin_ranking_loss(
        ref_input1, ref_input2, ref_target, margin, reduction
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.margin_ranking_loss(
            input1, input2, target, margin, reduction
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
