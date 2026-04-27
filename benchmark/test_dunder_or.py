import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.dunder_or
def test_dunder_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_or",
        torch_op=lambda a, b: a | b,
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
    )
    bench.run()
