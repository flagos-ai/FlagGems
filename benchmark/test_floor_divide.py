import pytest
import torch

from . import base, consts


@pytest.mark.floor_divide
def test_floor_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide",
        torch_op=torch.floor_divide,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.floor_divide_
def test_floor_divide_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="floor_divide_",
        torch_op=lambda a, b: a.floor_divide_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
