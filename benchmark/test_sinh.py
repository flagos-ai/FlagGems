import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sinh_
def test_sinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh_",
        torch_op=torch.sinh_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
