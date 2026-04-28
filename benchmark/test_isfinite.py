import pytest
import torch

from . import base, consts


@pytest.mark.isfinite
def test_isfinite():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isfinite", torch_op=torch.isfinite, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
