import pytest
import torch

from . import base, consts


@pytest.mark.sqrt
def test_sqrt():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sqrt", torch_op=torch.sqrt, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.sqrt_
def test_sqrt_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sqrt_",
        torch_op=torch.sqrt_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
