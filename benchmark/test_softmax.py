import pytest
import torch

from . import base, consts


@pytest.mark.softmax
def test_softmax():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softmax_backward
def test_softmax_backward():
    bench = base.UnaryReductionBenchmark(
        op_name="softmax",
        torch_op=torch.nn.functional.softmax,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()
