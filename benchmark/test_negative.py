import pytest
import torch

from . import base, consts


@pytest.mark.negative
def test_negative():
    bench = base.UnaryPointwiseBenchmark(
        op_name="negative", torch_op=torch.negative, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
