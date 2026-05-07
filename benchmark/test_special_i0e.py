import pytest
import torch

from . import base, consts


@pytest.mark.special_i0e
def test_special_i0e():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i0e",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
