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


@pytest.mark.mish_
def test_mish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mish_",
        torch_op=torch.ops.aten.mish_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
