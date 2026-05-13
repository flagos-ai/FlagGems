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
