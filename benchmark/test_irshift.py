import pytest
import torch

from . import base, consts


@pytest.mark.irshift__
def test_irshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="irshift__",
        torch_op=torch.ops.aten.__irshift__,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
