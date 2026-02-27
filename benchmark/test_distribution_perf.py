import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, unary_input_fn


def poisson_input_fn(shape, cur_dtype, device):
    # Poisson rate parameter (lambda), must be non-negative
    # Using values in range [0, 10] for reasonable sampling
    inp = torch.rand(shape, dtype=cur_dtype, device=device) * 10
    yield (inp,)


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


def normal__input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    loc = 3.0
    scale = 10.0
    yield self, loc, scale


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
        pytest.param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            marks=pytest.mark.exponential_,
        ),
        pytest.param(
            "poisson",
            torch.poisson,
            poisson_input_fn,
            marks=pytest.mark.poisson,
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
