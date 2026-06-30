import pytest
import torch

from . import base, consts


@pytest.mark.minimum
def test_minimum():
    bench = base.BinaryPointwiseBenchmark(
        op_name="minimum",
        torch_op=torch.minimum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
