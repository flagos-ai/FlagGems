import pytest
import torch

from . import base, consts


@pytest.mark.selu
def test_selu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="selu", torch_op=torch.nn.functional.selu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.selu_
def test_selu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="selu_",
        torch_op=torch.ops.aten.selu_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
