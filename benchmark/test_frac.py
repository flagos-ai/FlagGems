import pytest
import torch

from . import base, consts


@pytest.mark.frac
def test_frac():
    bench = base.UnaryPointwiseBenchmark(
        op_name="frac",
        torch_op=torch.frac,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
