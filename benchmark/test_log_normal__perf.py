import pytest
import torch

from benchmark.performance_utils import GenericBenchmark2DOnly


def log_normal_input_fn(shape, dtype, device):
    yield (torch.empty(shape, dtype=dtype, device=device),)


@pytest.mark.log_normal_
def test_perf_log_normal_():
    bench = GenericBenchmark2DOnly(
        op_name="log_normal_",
        torch_op=torch.Tensor.log_normal_,
        input_fn=log_normal_input_fn,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()
