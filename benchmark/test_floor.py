import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.floor_
def test_floor_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor_",
        torch_op=torch.Tensor.floor_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
