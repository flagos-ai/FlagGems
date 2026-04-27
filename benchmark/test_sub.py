import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sub
def test_sub():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.sub_
def test_sub_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub_",
        torch_op=lambda a, b: a.sub_(b),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
