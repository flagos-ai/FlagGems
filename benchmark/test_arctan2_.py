import pytest

from . import base, consts


@pytest.mark.arctan2_
def test_arctan2_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="arctan2_",
        torch_op=lambda a, b: a.arctan2_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
