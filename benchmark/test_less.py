import pytest
import torch

from . import base, consts


@pytest.mark.less
def test_less():
    bench = base.BinaryPointwiseBenchmark(
        op_name="less",
        torch_op=torch.less,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
