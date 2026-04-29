import pytest
import torch

from . import base, consts


@pytest.mark.acos
def test_acos():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acos", torch_op=torch.acos, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
