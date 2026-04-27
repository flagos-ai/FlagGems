import pytest

from . import attri_util as attrs
from . import performance_utils as base


@pytest.mark.dunder_ior
def test_dunder_ior_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_ior",
        torch_op=lambda a, b: a.__ior__(b),
        dtypes=attrs.INT_DTYPES + attrs.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
