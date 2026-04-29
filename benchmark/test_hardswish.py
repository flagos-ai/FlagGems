import pytest
import torch

from . import base, consts


@pytest.mark.hardswish_
def test_hardswish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish_",
        torch_op=torch.ops.aten.hardswish_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
