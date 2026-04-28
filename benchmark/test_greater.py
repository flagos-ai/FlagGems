import pytest
import torch

from . import base, consts


@pytest.mark.greater
def test_greater():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater",
        torch_op=torch.greater,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
