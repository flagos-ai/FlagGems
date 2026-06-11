import pytest
import torch

from . import base, consts


@pytest.mark.atanh
def test_atanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atanh",
        torch_op=torch.atanh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
