import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark


def log10_input_fn(shape, dtype, device):
    yield torch.rand(shape, dtype=dtype, device=device) + 0.01,


def log10_out_input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) + 0.01
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.log10
def test_log10():
    bench = GenericBenchmark(
        input_fn=log10_input_fn,
        op_name="log10",
        torch_op=torch.log10,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.log10_
def test_log10_inplace():
    bench = GenericBenchmark(
        input_fn=log10_input_fn,
        op_name="log10_",
        torch_op=torch.log10_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.log10_out
def test_log10_out():
    bench = GenericBenchmark(
        input_fn=log10_out_input_fn,
        op_name="log10_out",
        torch_op=torch.log10,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
