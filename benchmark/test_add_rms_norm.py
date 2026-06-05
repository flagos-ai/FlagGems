import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.add_rms_norm
def test_add_rms_norm():
    def add_rms_norm_input_fn(shape, dtype, device):
        M, N = shape
        inp1 = torch.randn(shape, dtype=dtype, device=device)
        inp2 = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield (inp1, inp2, (N,), weight)

    # Use a custom wrapper for torch implementation
    def torch_add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
        x = x1 + x2
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    bench = base.GenericBenchmark2DOnly(
        input_fn=add_rms_norm_input_fn,
        op_name="add_rms_norm",
        torch_op=torch_add_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.add_rms_norm)
    bench.run()
