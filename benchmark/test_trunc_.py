import pytest
import torch

from . import base, consts


@pytest.mark.trunc_
def test_trunc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="trunc_",
        torch_op=torch.trunc_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
