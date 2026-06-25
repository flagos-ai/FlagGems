import pytest
import torch

from . import base, consts


@pytest.mark.mish
def test_mish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mish",
        torch_op=torch.ops.aten.mish,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
