import pytest
import torch

from . import base, consts


@pytest.mark.arccos_
def test_arccos_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccos_",
        torch_op=torch.arccos_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
