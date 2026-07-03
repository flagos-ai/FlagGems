import pytest
import torch

from . import base, consts


@pytest.mark.arccos
def test_arccos():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccos",
        torch_op=torch.arccos,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.arccos_
def test_arccos_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccos_",
        torch_op=torch.arccos_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
