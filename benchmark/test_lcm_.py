import pytest

from . import base, consts


@pytest.mark.lcm_
def test_lcm_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lcm_",
        torch_op=lambda a, b: a.lcm_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
