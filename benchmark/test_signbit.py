import pytest
import torch

from .attri_util import FLOAT_DTYPES, INT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn


def signbit_out_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"out": torch.empty(shape, dtype=torch.bool, device=device)}


@pytest.mark.signbit
def test_signbit():
    bench = GenericBenchmark(
        input_fn=unary_input_fn,
        op_name="signbit",
        torch_op=torch.signbit,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )
    bench.run()


@pytest.mark.skip(reason="No support to non-boolean outputs: issue #2689.")
@pytest.mark.signbit_out
def test_signbit_out():
    bench = GenericBenchmark(
        input_fn=signbit_out_input_fn,
        op_name="signbit_out",
        torch_op=torch.signbit,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
