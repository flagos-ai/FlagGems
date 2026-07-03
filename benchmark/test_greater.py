import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, binary_input_fn, generate_tensor_input


def greater_out_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    yield inp1, inp2, {"out": torch.empty(shape, dtype=torch.bool, device=device)}


def greater_scalar_input_fn(shape, dtype, device):
    yield generate_tensor_input(shape, dtype, device), 0.5


def greater_scalar_out_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    yield inp, 0.5, {"out": torch.empty(shape, dtype=torch.bool, device=device)}


@pytest.mark.greater
def test_greater():
    bench = GenericBenchmark(
        input_fn=binary_input_fn,
        op_name="greater",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_out
def test_greater_out():
    bench = GenericBenchmark(
        input_fn=greater_out_input_fn,
        op_name="greater_out",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_scalar
def test_greater_scalar():
    bench = GenericBenchmark(
        input_fn=greater_scalar_input_fn,
        op_name="greater_scalar",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_scalar_out
def test_greater_scalar_out():
    bench = GenericBenchmark(
        input_fn=greater_scalar_out_input_fn,
        op_name="greater_scalar_out",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
