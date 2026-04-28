import pytest
import torch

from . import base, consts


@pytest.mark.lt
def test_lt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
