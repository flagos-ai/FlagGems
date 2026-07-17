import pytest

from . import base, consts


@pytest.mark.mvlgamma_
def test_mvlgamma_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mvlgamma_",
        torch_op=lambda a: a.mvlgamma_(5),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
