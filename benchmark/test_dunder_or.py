import pytest

from . import base, consts


@pytest.mark.dunder_or
def test_dunder_or():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_or",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
