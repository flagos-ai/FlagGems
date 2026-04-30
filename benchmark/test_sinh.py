import pytest
import torch

from . import base, consts


@pytest.mark.sinh_
def test_sinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh_",
        torch_op=torch.sinh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
