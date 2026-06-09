import pytest

from . import base, consts


@pytest.mark.arcsin_
def test_arcsin_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arcsin_",
        torch_op=lambda a: a.arcsin_(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
