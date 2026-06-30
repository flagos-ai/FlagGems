import pytest
import torch

from . import base, consts, utils


@pytest.mark.floor_divide
def test_floor_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide",
        torch_op=torch.floor_divide,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.floor_divide_
def test_floor_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_",
        torch_op=lambda a, b: a.floor_divide_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _floor_divide_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 3


@pytest.mark.floor_divide_scalar
def test_floor_divide_scalar():
    bench = base.GenericBenchmark(
        op_name="floor_divide_scalar",
        torch_op=torch.floor_divide,
        input_fn=_floor_divide_scalar_input_fn,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


def _floor_divide_scalar_inplace_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 3


@pytest.mark.floor_divide_scalar_
def test_floor_divide_scalar_():
    bench = base.GenericBenchmark(
        op_name="floor_divide_scalar_",
        torch_op=lambda a, b: a.floor_divide_(b),
        input_fn=_floor_divide_scalar_inplace_input_fn,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.floor_divide_tensor
def test_floor_divide_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_tensor",
        torch_op=torch.floor_divide,
        dtypes=[torch.float32] + consts.INT_DTYPES,
    )
    bench.run()
