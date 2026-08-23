import pytest
import torch

from . import base, consts


@pytest.mark.acos_
def test_acos_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acos_",
        torch_op=torch.acos_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
