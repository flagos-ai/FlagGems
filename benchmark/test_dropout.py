import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.dropout
def test_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="dropout", torch_op=torch.nn.Dropout(p=0.5), dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()
