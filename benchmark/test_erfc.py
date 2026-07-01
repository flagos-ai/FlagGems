import pytest
import torch

from . import base, consts


@pytest.mark.erfc
def test_erfc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfc",
        torch_op=torch.erfc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.erfc_
def test_erfc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfc_",
        torch_op=torch.erfc_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
