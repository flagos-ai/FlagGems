import pytest

from . import base, consts


@pytest.mark.dunder_ior_tensor
def test_dunder_ior_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_ior",
        torch_op=lambda a, b: a.__ior__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
