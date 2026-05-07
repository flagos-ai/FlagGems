import pytest
import torch

from . import base, consts


@pytest.mark.tanh
def test_tanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh", torch_op=torch.tanh, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.tanh_
def test_tanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh_",
        torch_op=torch.tanh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
