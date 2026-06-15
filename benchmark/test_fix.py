import pytest
import torch

from . import base, consts


@pytest.mark.fix
def test_fix():
    bench = base.UnaryPointwiseBenchmark(
        op_name="fix", torch_op=torch.fix, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
