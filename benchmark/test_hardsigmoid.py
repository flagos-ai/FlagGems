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


@pytest.mark.skip(reason="Hardsigmoid doesn't accept 'out': issue #2686")
@pytest.mark.hardsigmoid_out
def test_hardsigmoid_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="hardsigmoid_out",
        torch_op=torch.nn.functional.hardsigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.hardsigmoid_backward
def test_hardsigmoid_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardsigmoid_backward",
        torch_op=torch.ops.aten.hardsigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
