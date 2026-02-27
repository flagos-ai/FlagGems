import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
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


def log_normal__input_fn(shape, cur_dtype, device):
    self = torch.empty(shape, dtype=cur_dtype, device=device)
    mean = 1.0
    std = 2.0
    yield self, mean, std


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            marks=pytest.mark.exponential_,
        ),
        pytest.param(
            "log_normal_",
            torch.Tensor.log_normal_,
            log_normal__input_fn,
            marks=pytest.mark.log_normal_,
        ),
        pytest.param(
            "normal",
            torch.normal,
            normal_input_fn,
            marks=pytest.mark.normal,
        ),
        pytest.param(
            "normal_",
            torch.Tensor.normal_,
            normal__input_fn,
            marks=pytest.mark.normal_,
        ),
        pytest.param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            marks=pytest.mark.uniform_,
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
