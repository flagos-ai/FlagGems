import pytest
import torch

from . import base, consts


@pytest.mark.float_power
def test_float_power():
    bench = base.BinaryPointwiseBenchmark(
        op_name="float_power",
        torch_op=torch.float_power,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
