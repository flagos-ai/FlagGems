import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.alias
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_alias(shape, dtype):
    """Test alias accuracy: returned tensor shares storage with input."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.alias.default(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.alias.default(inp)

    # Values must match exactly (same storage)
    utils.gems_assert_close(utils.to_reference(res_out), ref_out, dtype)

    # Verify it's a view (shares storage)
    assert res_out.data_ptr() == inp.data_ptr()
