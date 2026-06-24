import pytest
import torch

from . import base, consts


@pytest.mark.erfinv
def test_erfinv():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfinv", torch_op=torch.erfinv, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.erfinv_
def test_erfinv_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfinv_",
        torch_op=lambda a: a.erfinv_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
