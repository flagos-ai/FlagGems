import pytest
import torch

from . import base, consts


@pytest.mark.logical_or
def test_logical_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_or",
        torch_op=torch.logical_or,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.logical_or_
def test_logical_or_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_or_",
        torch_op=lambda a, b: a.logical_or_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
