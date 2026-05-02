import pytest
import torch

from . import base, consts


@pytest.mark.bitwise_and_tensor
def test_bitwise_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_and_tensor_
def test_bitwise_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
