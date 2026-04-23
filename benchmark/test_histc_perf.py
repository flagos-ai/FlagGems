import pytest
import torch

from benchmark.performance_utils import GenericBenchmark2DOnly


def histc_input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) * 10
    yield inp, {"bins": 100, "min": 0, "max": 10}


@pytest.mark.histc
def test_perf_histc():
    bench = GenericBenchmark2DOnly(
        input_fn=histc_input_fn,
        op_name="histc",
        torch_op=torch.histc,
        dtypes=[torch.float32],
    )
    bench.run()
