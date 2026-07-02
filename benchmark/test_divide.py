import pytest
import torch

from . import base, consts


@pytest.mark.divide
def test_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="divide",
        torch_op=torch.divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.divide_
def test_divide_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="divide_",
        torch_op=lambda a, b: a.divide_(b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
