import pytest

from . import base, consts


@pytest.mark.gcd_
def test_gcd_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gcd_",
        torch_op=lambda a, b: a.gcd_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
