import pytest
import torch

from . import base, consts


@pytest.mark.logical_and
def test_logical_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_and",
        torch_op=torch.logical_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.logical_and_
def test_logical_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_and_",
        torch_op=lambda a, b: a.logical_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
