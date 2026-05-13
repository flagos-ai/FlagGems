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
    yield inp, 0x5A


@pytest.mark.bitwise_or_scalar_
def test_bitwise_or_scalar_():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="bitwise_or_scalar_",
        torch_op=torch.Tensor.bitwise_or_,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        inplace=True,
    )
    bench.run()
