import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.log1p_
def test_log1p_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log1p_",
        torch_op=lambda a: a.log1p_(),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
