import pytest
import torch

from . import base, consts


@pytest.mark.hypot
def test_hypot():
    bench = base.BinaryPointwiseBenchmark(
        op_name="hypot",
        torch_op=torch.hypot,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
