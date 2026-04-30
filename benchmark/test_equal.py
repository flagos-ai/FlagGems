import pytest
import torch

from . import base, consts


@pytest.mark.equal
def test_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="equal",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
