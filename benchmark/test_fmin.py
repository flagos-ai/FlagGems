import pytest
import torch

from . import base, consts


@pytest.mark.fmin
def test_fmin():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmin",
        torch_op=torch.fmin,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
