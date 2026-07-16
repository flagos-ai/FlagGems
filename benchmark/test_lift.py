import pytest
import torch

from . import base, consts


@pytest.mark.lift
def test_lift():
    bench = base.UnaryPointwiseBenchmark(
        op_name="lift",
        torch_op=torch.ops.aten.lift,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lift_out
def test_lift_out():
    # Uses UnaryPointwiseOutBenchmark (passes out= via get_input_iter)
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="lift_out",
        torch_op=torch.ops.aten.lift.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
