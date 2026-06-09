import pytest

from . import base, consts


@pytest.mark.sgn_
def test_sgn_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sgn_",
        torch_op=lambda a: a.sgn_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
