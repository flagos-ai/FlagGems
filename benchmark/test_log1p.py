import pytest

from . import base, consts


@pytest.mark.log1p_
def test_log1p_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log1p_",
        torch_op=lambda a: a.log1p_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
