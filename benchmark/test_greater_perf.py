import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def greater_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


def greater_scalar_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 0


@pytest.mark.greater
def test_perf_greater():
    bench = GenericBenchmark(
        input_fn=greater_input_fn,
        op_name="greater",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater
def test_perf_greater_scalar():
    bench = GenericBenchmark(
        input_fn=greater_scalar_input_fn,
        op_name="greater.Scalar",
        torch_op=torch.greater,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
