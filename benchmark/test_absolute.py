import pytest
import torch

from . import base, consts


@pytest.mark.absolute
def test_absolute():
    bench = base.UnaryPointwiseBenchmark(
        op_name="absolute", torch_op=torch.absolute, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
