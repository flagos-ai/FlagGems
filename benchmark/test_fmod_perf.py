import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def fmod_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    yield inp1, inp2


def fmod_scalar_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    yield inp1, 0.5


@pytest.mark.fmod
def test_fmod():
    bench = GenericBenchmark(
        input_fn=fmod_input_fn,
        op_name="fmod",
        torch_op=torch.fmod,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod
def test_fmod_scalar():
    bench = GenericBenchmark(
        input_fn=fmod_scalar_input_fn,
        op_name="fmod.Scalar",
        torch_op=torch.fmod,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod
def test_fmod_inplace():
    bench = GenericBenchmark(
        input_fn=fmod_input_fn,
        op_name="fmod_",
        torch_op=torch.Tensor.fmod_,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod
def test_fmod_inplace_scalar():
    bench = GenericBenchmark(
        input_fn=fmod_scalar_input_fn,
        op_name="fmod_.Scalar",
        torch_op=torch.Tensor.fmod_,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
