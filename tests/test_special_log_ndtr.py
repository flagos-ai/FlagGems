import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_log_ndtr
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special_log_ndtr only supports float32 (PyTorch aten::special_log_ndtr has no Half/BFloat16 dispatch)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_log_ndtr(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    # Reference: call outside use_gems to get PyTorch's implementation
    ref_out = torch.ops.aten.special_log_ndtr(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_log_ndtr(inp)

    # Allow for numerical differences between our erf-based implementation
    # and PyTorch's implementation (which uses a different algorithm)
    utils.gems_assert_close(res_out, ref_out, dtype)
