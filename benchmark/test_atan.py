import pytest
import torch

from . import base, consts


@pytest.mark.atan
def test_atan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan", torch_op=torch.atan, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.atan_
def test_atan_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan_",
        torch_op=torch.atan_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
