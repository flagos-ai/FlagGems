import pytest
import torch

from . import base, consts


@pytest.mark.gelu
def test_gelu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu", torch_op=torch.nn.functional.gelu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.gelu_
def test_gelu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu_",
        torch_op=torch.ops.aten.gelu_.default,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.gelu_backward
def test_gelu_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu_backward",
        torch_op=torch.nn.functional.gelu,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
