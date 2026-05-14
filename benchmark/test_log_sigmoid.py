import pytest
import torch

from . import base, consts


@pytest.mark.log_sigmoid
def test_log_sigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log_sigmoid",
        torch_op=torch.nn.functional.logsigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
