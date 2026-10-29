import pytest
import torch

from . import base, consts


@pytest.mark.bitwise_right_shift_
def test_irshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_right_shift_",
        torch_op=torch.ops.aten.__irshift__,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
