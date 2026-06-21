import pytest

from . import base, consts


@pytest.mark.ilshift
def test_ilshift():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ilshift",
        torch_op=lambda a, b: a.__ilshift__(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
