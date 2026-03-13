import os
import sys
import pytest
import torch

import flag_gems

from flag_gems.experimental_ops.select_backward import (
    select_backward as gems_select_backward,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "shape",
    [
        (4, 8, 16),
        (2, 3, 4, 5),
        (8, 16, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_select_backward_accuracy(shape, dtype, dim):

    device = flag_gems.device

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    dim = dim if dim >= 0 else dim + len(shape)

    index = shape[dim] // 2

    y = torch.select(x, dim, index)

    grad = torch.randn_like(y)

    y.backward(grad)

    ref_grad = x.grad

    with flag_gems.use_gems():

        act_grad = gems_select_backward(
            grad,
            x.shape,
            dim,
            index,
        )

    torch.testing.assert_close(act_grad, ref_grad, rtol=1e-3, atol=1e-3)