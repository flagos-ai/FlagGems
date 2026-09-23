import pytest
import torch

from . import base, consts


@pytest.mark.isposinf
def test_isposinf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isposinf",
        torch_op=torch.isposinf,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
