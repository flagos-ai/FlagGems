import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes from the worktree PDIST_SHAPES covering small to large matrices
PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward
    p = 2.0
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    # Compute backward with gradient of ones
    grad_output = torch.ones_like(pdist_out)

    ref_out = torch.ops.aten._pdist_backward(grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward with p=1
    p = 1.0
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    # Compute backward with gradient of ones
    grad_output = torch.ones_like(pdist_out)

    ref_out = torch.ops.aten._pdist_backward(grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)
