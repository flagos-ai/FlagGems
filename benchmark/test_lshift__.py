import pytest
import torch

from . import base, consts


@pytest.mark.lshift__
def test_lshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lshift__",
        torch_op=torch.bitwise_left_shift,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
