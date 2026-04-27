import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.arctanh_
def test_arctanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctanh_",
        torch_op=lambda a: a.arctanh_(),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
