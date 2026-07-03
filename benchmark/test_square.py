import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def square_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.square
def test_square():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="square",
        torch_op=torch.square,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.square_
def test_square_inplace():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="square_",
        torch_op=torch.square_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.square_out
def test_square_out():
    bench = GenericBenchmark(
        input_fn=square_out_input_fn,
        op_name="square_out",
        torch_op=torch.square,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
