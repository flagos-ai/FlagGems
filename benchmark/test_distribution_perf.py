import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


def normal__input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


def random__input_fn(shape, cur_dtype, device):
    self = torch.empty(shape, dtype=cur_dtype, device=device)
    yield self, 3, 97


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "normal",
            torch.normal,
            normal_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.normal,
        ),
        pytest.param(
            "normal_",
            torch.Tensor.normal_,
            normal__input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.normal_,
        ),
        pytest.param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.uniform_,
        ),
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.exponential_,
        ),
        pytest.param(
            "random_",
            torch.Tensor.random_,
            random__input_fn,
            INT_DTYPES,
            marks=pytest.mark.random_,
        ),
    ],
)
def test_distribution_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
    )
    bench.run()
