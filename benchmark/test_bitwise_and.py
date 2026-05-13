import pytest
import torch

from . import base, consts


@pytest.mark.bitwise_and_tensor
def test_bitwise_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_and_tensor_
def test_bitwise_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_input_fn(shape, cur_dtype, device):
    scalar = 0x96 if cur_dtype != torch.bool else True
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    yield scalar, inp


@pytest.mark.bitwise_and_scalar_tensor
def test_bitwise_and_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_tensor",
        torch_op=torch.bitwise_and,
        input_fn=scalar_tensor_input_fn,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
