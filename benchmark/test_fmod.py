import pytest
import torch

from . import base, consts, utils


def _tensor_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    yield inp1, inp2


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.fmod_tensor
def test_fmod_tensor():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="fmod_tensor",
        torch_op=torch.fmod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod_scalar
def test_fmod_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="fmod_scalar",
        torch_op=torch.fmod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod_tensor_
def test_fmod_tensor_():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="fmod_tensor_",
        torch_op=torch.Tensor.fmod_,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.fmod_scalar_
def test_fmod_scalar_():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="fmod_scalar_",
        torch_op=torch.Tensor.fmod_,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()
