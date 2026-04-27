import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.pow
def test_pow():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=torch.pow,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_
def test_pow_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
