import pytest
import torch

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_out_input_fn(shape, dtype, device):
    inp1 = base.generate_tensor_input(shape, dtype, device)
    inp2 = base.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.greater_out
def test_greater_out():
    bench = base.GenericBenchmark(
        op_name="greater_out",
        torch_op=torch.greater,
        input_fn=greater_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_scalar
def test_greater_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="greater_scalar",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def greater_scalar_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=torch.bool, device=device)
    yield inp, 0, {"out": out}


@pytest.mark.greater_scalar_out
def test_greater_scalar_out():
    bench = base.GenericBenchmark(
        op_name="greater_scalar_out",
        input_fn=greater_scalar_out_input_fn,
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
