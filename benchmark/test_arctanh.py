import pytest

from . import base, consts


@pytest.mark.arctanh_
def test_arctanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctanh_",
        torch_op=lambda a: a.arctanh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
