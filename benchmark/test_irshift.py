import pytest
import torch

from . import base, consts


@pytest.mark.irshift
def test_irshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="irshift",
        torch_op=torch.ops.aten.__irshift__,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
