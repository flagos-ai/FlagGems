import pytest
import torch

from . import base, consts


@pytest.mark.floor_
def test_floor_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor_",
        torch_op=torch.Tensor.floor_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
