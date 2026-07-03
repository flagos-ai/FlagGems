import pytest
import torch

from . import base, consts


@pytest.mark.subtract
def test_subtract():
    bench = base.BinaryPointwiseBenchmark(
        op_name="subtract",
        torch_op=torch.subtract,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.subtract_
def test_subtract_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="subtract_",
        torch_op=lambda a, b: a.subtract_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
