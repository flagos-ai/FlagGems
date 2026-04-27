import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.digamma_
def test_digamma_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="digamma_",
        torch_op=lambda a: a.digamma_(),
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
