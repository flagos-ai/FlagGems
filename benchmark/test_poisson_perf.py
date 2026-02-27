import pytest
import torch

from benchmark.performance_utils import GenericBenchmark2DOnly


def poisson_input_fn(shape, dtype, device):
    yield (torch.rand(shape, dtype=dtype, device=device) * 10,)


@pytest.mark.poisson
def test_perf_poisson():
    bench = GenericBenchmark2DOnly(
        op_name="poisson",
        torch_op=torch.poisson,
        input_fn=poisson_input_fn,
        dtypes=[torch.float32],
    )
    bench.run()
