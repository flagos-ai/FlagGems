import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.select_backward import \
    select_backward as gems_select_backward

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (4, 8),
        (4, 8, 16),
        (2, 3, 4, 5),
        (8, 16, 32),
        (3, 7, 11),
        (2, 1, 4),
        (64, 512),
        (32, 256, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_select_backward_accuracy(shape, dtype, dim):
    device = flag_gems.device

    ndim = len(shape)

    actual_dim = dim + ndim if dim < 0 else dim

    if actual_dim >= ndim:
        pytest.skip(f"dim {dim} out of range for shape {shape}")

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    dim_size = shape[actual_dim]
    indices_to_test = [0, dim_size // 2]
    if dim_size > 1:
        indices_to_test.append(dim_size - 1)

    for index in indices_to_test:
        if x.grad is not None:
            x.grad.zero_()

        y = torch.select(x, actual_dim, index)
        grad = torch.randn_like(y)

        y.backward(grad)
        ref_grad = x.grad.clone()

        with flag_gems.use_gems():
            act_grad = gems_select_backward(
                grad,
                x.shape,
                actual_dim,
                index,
            )

        assert act_grad.shape == ref_grad.shape
        assert act_grad.dtype == ref_grad.dtype

        torch.testing.assert_close(act_grad, ref_grad, rtol=0, atol=0)
