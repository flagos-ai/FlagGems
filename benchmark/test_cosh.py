import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def cosh_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.cosh
def test_cosh():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="cosh",
        torch_op=torch.cosh,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.cosh_
def test_cosh_inplace():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="cosh_",
        torch_op=torch.cosh_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.cosh_out
def test_cosh_out():
    bench = GenericBenchmark(
        input_fn=cosh_out_input_fn,
        op_name="cosh_out",
        torch_op=torch.cosh,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
