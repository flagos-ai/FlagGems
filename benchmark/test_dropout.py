import pytest
import torch

from . import base, consts


@pytest.mark.dropout
def test_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="dropout", torch_op=torch.nn.Dropout(p=0.5), dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
