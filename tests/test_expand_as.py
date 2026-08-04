import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.expand_as
@pytest.mark.parametrize(
    "shape,target_shape",
    [
        ((1,), (64,)),
        ((1, 64), (128, 64)),
        ((1, 1, 512), (64, 512, 512)),
        ((4096, 1), (4096, 4096)),
        ((1, 1, 1, 64), (8, 16, 32, 64)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_expand_as(shape, target_shape, dtype):
    """Test expand_as accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = ref_inp.expand_as(ref_other)
    with flag_gems.use_gems():
        res_out = inp.expand_as(other)

    utils.gems_assert_close(utils.to_reference(res_out), ref_out, dtype)
