import pytest
import torch

from . import base, consts


@pytest.mark.add
def test_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.add_
def test_add_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_",
        torch_op=lambda a, b: a.add_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
