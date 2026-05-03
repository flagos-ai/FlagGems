import pytest
import torch

from . import base


@pytest.mark.rms_norm
def test_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        _, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = base.GenericBenchmark2DOnly(
        op_name="rms_norm",
        input_fn=rms_norm_input_fn,
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()
