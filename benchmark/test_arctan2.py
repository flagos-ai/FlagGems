import pytest
import torch

from . import base, consts


@pytest.mark.arctan2
def test_arctan2():
    bench = base.BinaryPointwiseBenchmark(
        op_name="arctan2",
        torch_op=torch.arctan2,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
