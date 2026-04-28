import pytest
import torch

from . import base, consts


@pytest.mark.logical_xor
def test_logical_xor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_xor",
        torch_op=torch.logical_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
