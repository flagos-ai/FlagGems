import pytest
import torch

from . import base, consts


@pytest.mark.frexp
def test_frexp():
    bench = base.UnaryPointwiseBenchmark(
        op_name="frexp",
        torch_op=torch.frexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
