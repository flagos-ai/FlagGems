import pytest
import torch

from . import base, consts


@pytest.mark.lshift
def test_lshift():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lshift",
        torch_op=torch.bitwise_left_shift,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
