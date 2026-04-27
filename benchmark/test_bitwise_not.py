import pytest
import torch

from . import base, consts


@pytest.mark.bitwise_not
def test_bitwise_not():
    bench = base.UnaryPointwiseBenchmark(
        op_name="bitwise_not", torch_op=torch.bitwise_not, dtypes=consts.INT_DTYPES
    )
    bench.run()


def test_bitwise_not_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="bitwise_not_",
        torch_op=lambda a: a.bitwise_not_(),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
