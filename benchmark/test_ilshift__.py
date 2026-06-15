import pytest

from . import base, consts


@pytest.mark.ilshift__
def test_ilshift__():
    bench = base.BinaryPointwiseBenchmark(
        op_name="ilshift__",
        torch_op=lambda a, b: a.__ilshift__(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
