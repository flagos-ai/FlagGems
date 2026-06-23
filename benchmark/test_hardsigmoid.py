import pytest
import torch

from . import base, consts


@pytest.mark.hardsigmoid
def test_hardsigmoid():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardsigmoid",
        torch_op=torch.nn.functional.hardsigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.hardsigmoid_out
def test_hardsigmoid_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="hardsigmoid_out",
        torch_op=torch.ops.aten.hardsigmoid.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
