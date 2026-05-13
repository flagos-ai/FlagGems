import pytest
import torch

from . import base, consts


@pytest.mark.atan2
def test_atan2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="atan2",
        torch_op=torch.atan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
