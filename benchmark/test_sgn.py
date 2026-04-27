import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.sgn_
def test_sgn_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atan_",
        torch_op=lambda a: a.sgn_(),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
