import pytest
import torch

from . import base, consts


@pytest.mark.lgamma
def test_lgamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="lgamma", torch_op=torch.lgamma, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.lgamma_
def test_lgamma_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="lgamma_",
        torch_op=lambda a: a.lgamma_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
