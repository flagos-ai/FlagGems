import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def expm1_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.expm1
def test_expm1():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="expm1",
        torch_op=torch.expm1,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.expm1_
def test_expm1_inplace():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="expm1_",
        torch_op=torch.expm1_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.expm1_out
def test_expm1_out():
    bench = GenericBenchmark(
        input_fn=expm1_out_input_fn,
        op_name="expm1_out",
        torch_op=torch.expm1,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
