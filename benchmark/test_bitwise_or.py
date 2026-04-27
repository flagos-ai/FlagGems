import pytest
import torch

from . import base, consts


@pytest.mark.bitwise_or
def test_bitwise_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or",
        torch_op=torch.bitwise_or,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_or_
def test_bitwise_or_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_or_",
        torch_op=lambda a, b: a.bitwise_or_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
