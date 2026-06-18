import pytest
import torch

from . import base, consts


@pytest.mark.logical_not
def test_logical_not():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logical_not", torch_op=torch.logical_not, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.logical_not_
def test_logical_not_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logical_not_",
        torch_op=lambda a: a.logical_not_(),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
