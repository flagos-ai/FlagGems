import pytest
import torch

from . import base, consts


@pytest.mark.special_xlog1py
def test_special_xlog1py():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_xlog1py",
        torch_op=torch.ops.aten.special_xlog1py,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
