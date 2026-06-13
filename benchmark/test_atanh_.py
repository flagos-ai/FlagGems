import pytest

from . import base, consts


@pytest.mark.atanh_
def test_atanh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atanh_",
        torch_op=lambda a: a.atanh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
