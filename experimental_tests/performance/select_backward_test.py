import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.select_backward import \
    select_backward as gems_select_backward

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "shape",
    [
        (128, 256),
        (64, 128, 256),
        (32, 64, 128, 256),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_select_backward_benchmark(shape, dtype, dim):

    device = flag_gems.device

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    dim = dim if dim >= 0 else dim + len(shape)

    index = shape[dim] // 2

    y = torch.select(x, dim, index)

    grad = torch.randn_like(y)

    quantiles = [0.5, 0.2, 0.8]

    def torch_impl():
        x.grad = None
        y = torch.select(x, dim, index)
        y.backward(grad)

    ms_torch, _, _ = triton.testing.do_bench(
        torch_impl,
        rep=100,
        quantiles=quantiles,
    )

    with flag_gems.use_gems():

        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_select_backward(grad, x.shape, dim, index),
            rep=100,
            quantiles=quantiles,
        )

    speedup = ms_torch / ms_triton

    print(f"select_backward {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
