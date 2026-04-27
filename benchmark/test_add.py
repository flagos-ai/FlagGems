import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.add
def test_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.add_
def test_add_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_",
        torch_op=lambda a, b: a.add_(b),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
