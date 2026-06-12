import pytest

from . import base, consts


@pytest.mark.fmod_
def test_fmod_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmod_",
        torch_op=lambda a, b: a.fmod_(b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
