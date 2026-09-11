import pytest
import torch

from . import base, consts


@pytest.mark.chalf
def test_chalf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="chalf",
        torch_op=torch.Tensor.chalf,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
