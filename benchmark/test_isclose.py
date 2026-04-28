import pytest
import torch

from . import base, consts


@pytest.mark.isclose
def test_isclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="isclose",
        torch_op=torch.isclose,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
