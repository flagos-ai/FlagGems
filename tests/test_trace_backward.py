import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Core shapes exercised by TRACE_BACKWARD_SHAPES (mirrors worktree CI branch).
TRACE_BACKWARD_SHAPES = [
    (1, 1),
    (3, 3),
    (4, 3),
    (3, 4),
    (128, 128),
    (200, 100),
    (100, 200),
    (1024, 1024),
    (1, 1000),
    (1000, 1),
]


@pytest.mark.trace_backward
@pytest.mark.parametrize("shape", TRACE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trace_backward(shape, dtype):
    res_grad = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)

    ref_out = torch.ops.aten.trace_backward(ref_grad, list(shape))
    res_out = flag_gems.ops.trace_backward(res_grad, list(shape))

    utils.gems_assert_equal(res_out, ref_out)
