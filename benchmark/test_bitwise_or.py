import pytest
import torch

from . import base, consts, utils


@pytest.mark.bitwise_or_tensor
def test_bitwise_or_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or_tensor",
        torch_op=torch.bitwise_or,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_or_tensor_
def test_bitwise_or_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or_tensor_",
        torch_op=lambda a, b: a.bitwise_or_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    scalar = True if dtype == torch.bool else 0x00FF
    yield inp, scalar


@pytest.mark.bitwise_or_scalar
def test_bitwise_or_scalar():
    bench = base.GenericBenchmark(
        op_name="bitwise_or_scalar",
        torch_op=torch.bitwise_or,
        input_fn=_scalar_input_fn,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def _scalar_input_fn_inplace(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0x5A


@pytest.mark.bitwise_or_scalar_
def test_bitwise_or_scalar_():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn_inplace,
        op_name="bitwise_or_scalar_",
        torch_op=lambda a, b: a.bitwise_or_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_input_fn(shape, cur_dtype, device):
    scalar = 0x00FF if cur_dtype != torch.bool else True
    tensor = utils.generate_tensor_input(shape, cur_dtype, device)
    yield scalar, tensor


@pytest.mark.bitwise_or_scalar_tensor
def test_bitwise_or_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="bitwise_or_scalar_tensor",
        torch_op=torch.bitwise_or,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        input_fn=scalar_tensor_input_fn,
    )
    bench.run()
