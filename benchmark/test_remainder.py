import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.remainder
def test_remainder():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder",
        torch_op=torch.remainder,
        dtypes=attrs.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.remainder_
def test_remainder_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=attrs.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
