import pytest
import torch

from . import base, consts


@pytest.mark.softplus
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
