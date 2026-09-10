import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unfold
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "params",
    [
        {"dimension": 0, "size": 4, "step": 2},
        {"dimension": 0, "size": 8, "step": 4},
        {"dimension": -1, "size": 4, "step": 1},
    ],
)
def test_unfold(shape, dtype, params):
    """Test unfold accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    dimension = params["dimension"]
    size = params["size"]
    step = params["step"]

    ref_out = ref_inp.unfold(dimension, size, step)
    with flag_gems.use_gems():
        res_out = inp.unfold(dimension, size, step)

    utils.gems_assert_close(utils.to_reference(res_out), ref_out, dtype)
