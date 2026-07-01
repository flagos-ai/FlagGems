import pytest

from . import base, consts


@pytest.mark.frac_
def test_frac_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="frac_",
        torch_op=lambda a: a.frac_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
