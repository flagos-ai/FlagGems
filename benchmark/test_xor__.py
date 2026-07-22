import pytest
import torch

from . import base, consts


@pytest.mark.xor__
def test_xor__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="xor__",
        torch_op=torch.bitwise_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
