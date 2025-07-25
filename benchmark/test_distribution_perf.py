import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "normal",
            torch.normal,
            normal_input_fn,
            marks=pytest.mark.normal,
        ),
        pytest.param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            marks=pytest.mark.uniform_,
        ),
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            marks=pytest.mark.exponential_,
        ),
    ],
)
def test_distribution_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
