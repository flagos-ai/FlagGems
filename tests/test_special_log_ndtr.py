import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_log_ndtr
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# torch.ops.aten.special_log_ndtr does not support float16/bfloat16 on CUDA
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_log_ndtr(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Reference: call outside use_gems to get PyTorch's implementation
    ref_out = torch.ops.aten.special_log_ndtr(utils.to_reference(inp))
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_log_ndtr(inp)

    # Allow for numerical differences between our erf-based implementation
    # and PyTorch's implementation (which uses a different algorithm)
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.2)
