import pytest

from . import base, consts


@pytest.mark.deg2rad_
def test_deg2rad_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="deg2rad_",
        torch_op=lambda a: a.deg2rad_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
