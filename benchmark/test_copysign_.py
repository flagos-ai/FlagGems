import pytest

from . import base, consts


@pytest.mark.copysign_
def test_copysign_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="copysign_",
        torch_op=lambda a, b: a.copysign_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
