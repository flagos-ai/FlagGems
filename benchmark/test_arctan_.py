import pytest
import torch

from . import base, consts


@pytest.mark.arctan
def test_arctan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctan",
        torch_op=torch.arctan,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.arctan_
def test_arctan_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctan_",
        torch_op=torch.arctan_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
