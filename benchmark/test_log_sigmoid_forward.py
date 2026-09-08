import pytest
import torch

from . import base, consts


@pytest.mark.log_sigmoid_forward
def test_log_sigmoid_forward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log_sigmoid_forward",
        torch_op=lambda x: torch.ops.aten.log_sigmoid_forward(x)[0],
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
