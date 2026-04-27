import pytest
import torch

from . import base, consts


@pytest.mark.angle
def test_angle():
    bench = base.UnaryPointwiseBenchmark(
        op_name="angle",
        torch_op=torch.angle,
        dtypes=consts.COMPLEX_DTYPES
        + [torch.float32]
        + consts.INT_DTYPES
        + consts.BOOL_DTYPES,
    )
    bench.run()
