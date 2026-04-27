import pytest
import torch

from . import base, consts


@pytest.mark.eq
def test_eq():
    bench = base.BinaryPointwiseBenchmark(
        op_name="eq",
        torch_op=torch.eq,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
