import pytest
import torch

from . import base, consts


@pytest.mark.isinf
def test_isinf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isinf", torch_op=torch.isinf, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
