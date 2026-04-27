import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.asinh
def test_asinh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="asinh", torch_op=torch.asinh, dtypes=attrs.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.asinh_
def test_asinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="asinh_",
        torch_op=lambda a: a.asinh_(),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
