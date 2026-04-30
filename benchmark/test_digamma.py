import pytest

from . import base, consts


@pytest.mark.digamma_
def test_digamma_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="digamma_",
        torch_op=lambda a: a.digamma_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
