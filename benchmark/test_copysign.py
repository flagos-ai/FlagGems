import pytest
import torch

from . import base, consts


@pytest.mark.copysign
def test_copysign():
    bench = base.BinaryPointwiseBenchmark(
        op_name="copysign",
        torch_op=torch.copysign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
