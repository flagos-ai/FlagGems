import pytest
import torch

from . import base, consts


@pytest.mark.greater_equal
def test_greater_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater_equal",
        torch_op=torch.greater_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
