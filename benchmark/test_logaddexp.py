import pytest
import torch

from . import base, consts


@pytest.mark.logaddexp
def test_logaddexp():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
