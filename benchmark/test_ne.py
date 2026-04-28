import pytest
import torch

from . import base, consts


@pytest.mark.ne
def test_ne():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ne",
        torch_op=torch.ne,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
