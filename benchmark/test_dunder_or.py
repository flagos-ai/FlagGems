import pytest

from . import base, consts


@pytest.mark.dunder_or_tensor
def test_dunder_or_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_or_tensor",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
