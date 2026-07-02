import pytest
import torch

from . import base, consts


@pytest.mark.mish_
def test_mish_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mish_",
        torch_op=torch.ops.aten.mish_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
