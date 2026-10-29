import pytest
import torch

from . import base, consts


@pytest.mark.underscore_add_relu
def test_add_relu():
    bench = base.BinaryPointwiseBenchmark(
        op_name="_add_relu",
        torch_op=lambda a, b: torch.relu(a + b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
