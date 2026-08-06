import pytest
import torch

from . import base, consts


@pytest.mark.special_ndtr
def test_special_ndtr():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_ndtr",
        torch_op=torch.ops.aten.special_ndtr,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
