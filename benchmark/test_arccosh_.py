import pytest

from . import base, consts


@pytest.mark.arccosh_
def test_arccosh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arccosh_",
        torch_op=lambda a: a.arccosh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
