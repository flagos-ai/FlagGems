import pytest
import torch

from . import base, consts


@pytest.mark.deg2rad
def test_deg2rad():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad", torch_op=torch.deg2rad, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
