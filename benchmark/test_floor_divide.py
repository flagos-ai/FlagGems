import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.floor_divide
def test_floor_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide",
        torch_op=torch.floor_divide,
        dtypes=attrs.INT_DTYPES,
    )
    bench.run()


@pytest.mark.floor_divide_
def test_floor_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_",
        torch_op=lambda a, b: a.floor_divide_(b),
        dtypes=attrs.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
