import pytest
import torch

from . import base, consts


@pytest.mark.logical_not
def test_logical_not():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logical_not", torch_op=torch.logical_not, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
