import pytest
import torch

from . import performance_utils as utils


@pytest.mark.rms_norm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.4"),
    reason="The version prior to 2.4 does not include the rms_norm API in torch.",
)
def test_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        _, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = utils.GenericBenchmark2DOnly(
        op_name="rms_norm",
        input_fn=rms_norm_input_fn,
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()
