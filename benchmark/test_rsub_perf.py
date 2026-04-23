import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def rsub_tensor_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, inp2


def rsub_scalar_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 0.5


@pytest.mark.rsub
def test_perf_rsub_tensor():
    bench = GenericBenchmark(
        input_fn=rsub_tensor_input_fn,
        op_name="rsub.Tensor",
        torch_op=torch.rsub,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rsub
def test_perf_rsub_scalar():
    bench = GenericBenchmark(
        input_fn=rsub_scalar_input_fn,
        op_name="rsub.Scalar",
        torch_op=torch.rsub,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
