import pytest
import torch

import flag_gems

from . import base

vendor_name = flag_gems.vendor_name


def weight_norm_input_fn(shape, dtype, device):
    dim = 0
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(len(shape))],
        dtype=dtype,
        device=device,
    )
    yield v, g, dim


def weight_norm_input_fn_last(shape, dtype, device):
    dim = len(shape) - 1
    v = torch.randn(shape, dtype=dtype, device=device)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(len(shape))],
        dtype=dtype,
        device=device,
    )
    yield v, g, dim


@pytest.mark.weight_norm
def test_weight_norm_dim0():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_dim0",
        input_fn=weight_norm_input_fn,
        torch_op=torch._weight_norm,
    )
    bench.set_gems(flag_gems.weight_norm)
    bench.run()


@pytest.mark.weight_norm
def test_weight_norm_dim_last():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_dim_last",
        input_fn=weight_norm_input_fn_last,
        torch_op=torch._weight_norm,
    )
    bench.set_gems(flag_gems.weight_norm)
    bench.run()
