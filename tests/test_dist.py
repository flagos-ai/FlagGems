import pytest
import torch

import flag_gems


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
    y = torch.randn(shape, device="cuda", dtype=dtype)

    with flag_gems.use_gems():
        out = torch.dist(x, y, p)

    ref = torch.dist(x, y, p)

    if dtype == torch.bfloat16:
        rtol = 1.6e-2
    else:
        rtol = 1e-3

    torch.testing.assert_close(
        out,
        ref,
        rtol=rtol,
        atol=1e-3,
    )


def test_dist_empty():
    x = torch.empty(0, device="cuda")
    y = torch.empty(0, device="cuda")

    with flag_gems.use_gems():
        out = torch.dist(x, y)

    ref = torch.dist(x, y)

    torch.testing.assert_close(out, ref)
