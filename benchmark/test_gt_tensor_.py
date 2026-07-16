import pytest

from . import base, consts


@pytest.mark.gt_tensor_
def test_gt_tensor_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt_tensor_",
        torch_op=lambda a, b: a.gt_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
