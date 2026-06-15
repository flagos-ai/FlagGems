import pytest
import torch

from . import base, consts


@pytest.mark.allclose
def test_allclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="allclose",
        torch_op=torch.allclose,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
