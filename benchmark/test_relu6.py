import pytest
import torch

from . import base, consts


@pytest.mark.relu6
def test_relu6():
    bench = base.UnaryPointwiseBenchmark(
        op_name="relu6", torch_op=torch.nn.functional.relu6, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
