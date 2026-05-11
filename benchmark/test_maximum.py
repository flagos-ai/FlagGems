import pytest
import torch

from . import base, consts


@pytest.mark.maximum
def test_maximum():
    bench = base.BinaryPointwiseBenchmark(
        op_name="maximum",
        torch_op=torch.maximum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
