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


@pytest.mark.softplus_backward
def test_softplus_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus_backward",
        torch_op=torch.ops.aten.softplus_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
