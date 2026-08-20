import pytest
import torch

from . import base, consts


@pytest.mark.lcm
def test_lcm():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm",
        torch_op=torch.lcm,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
