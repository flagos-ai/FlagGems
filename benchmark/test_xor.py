import pytest
import torch

from . import base, consts


@pytest.mark.xor
def test_xor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xor",
        torch_op=torch.bitwise_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
