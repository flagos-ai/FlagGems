import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.flatten
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512), (8, 16, 32, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "dims",
    [
        {"start_dim": 0, "end_dim": -1},
        {"start_dim": 0, "end_dim": 0},
        {"start_dim": 1, "end_dim": -1},
    ],
)
def test_flatten(shape, dtype, dims):
    """Test flatten accuracy against PyTorch implementation."""
    start_dim = dims["start_dim"]
    end_dim = dims["end_dim"]

    # Skip invalid combos (start_dim >= ndim)
    ndim = len(shape)
    if start_dim >= ndim:
        pytest.skip("start_dim >= ndim")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.flatten(ref_inp, start_dim, end_dim)
    with flag_gems.use_gems():
        res_out = torch.flatten(inp, start_dim, end_dim)

    utils.gems_assert_close(utils.to_reference(res_out), ref_out, dtype)
