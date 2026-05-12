import pytest
import torch

from . import base, consts


@pytest.mark.log
def test_log():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log", torch_op=torch.log, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
