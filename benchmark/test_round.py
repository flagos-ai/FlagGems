import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def round_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp,


def round_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.round
def test_round():
    bench = GenericBenchmark(
        input_fn=round_input_fn,
        op_name="round",
        torch_op=torch.round,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.round_
def test_round_inplace():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="round_",
        torch_op=torch.round_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.round_out
def test_round_out():
    bench = GenericBenchmark(
        input_fn=round_out_input_fn,
        op_name="round_out",
        torch_op=torch.round,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
