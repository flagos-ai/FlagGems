import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark


def bernoulli_inplace_input_fn(shape, dtype, device):
    yield torch.empty(shape, dtype=dtype, device=device), 0.5


def bernoulli_input_fn(shape, dtype, device):
    yield torch.rand(shape, dtype=dtype, device=device),


@pytest.mark.bernoulli_
def test_bernoulli_inplace():
    bench = GenericBenchmark(
        input_fn=bernoulli_inplace_input_fn,
        op_name="bernoulli_",
        torch_op=torch.Tensor.bernoulli_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.bernoulli
def test_bernoulli():
    bench = GenericBenchmark(
        input_fn=bernoulli_input_fn,
        op_name="bernoulli",
        torch_op=torch.bernoulli,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
