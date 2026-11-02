import pytest
import torch

from . import base, consts


@pytest.mark.ixor
def test_ixor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="__ixor__",
        torch_op=torch.ops.aten.__ixor__,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
