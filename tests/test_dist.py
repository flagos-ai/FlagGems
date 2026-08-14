import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.parametrize("p", [0, 1, 2, 3, float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "shape",
    [
        (1024,),
        (32, 1024),
        (4096,),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ],
)
def test_dist(shape, p, dtype):
    x = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    y = torch.randn(shape, device=flag_gems.device, dtype=dtype)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.dist(ref_x, ref_y, p)

    with flag_gems.use_gems():
        out = torch.dist(x, y, p)

    utils.gems_assert_close(out, ref_out, dtype)


def test_dist_empty():
    x = torch.empty(0, device=flag_gems.device)
    y = torch.empty(0, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)
    ref_out = torch.dist(ref_x, ref_y)

    with flag_gems.use_gems():
        out = torch.dist(x, y)

    utils.gems_assert_close(out, ref_out, x.dtype)
