import pytest
import torch

from . import base, consts


@pytest.mark.le
def test_le():
    bench = base.BinaryPointwiseBenchmark(
        op_name="le",
        torch_op=torch.le,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
