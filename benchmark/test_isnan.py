import pytest
import torch

from . import base, consts


@pytest.mark.isnan
def test_isnan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isnan", torch_op=torch.isnan, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
