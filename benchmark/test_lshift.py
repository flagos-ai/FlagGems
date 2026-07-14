import pytest
import torch

from . import base, consts


@pytest.mark.lshift
def test_lshift():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lshift",
        torch_op=torch.ops.aten.__lshift__,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
