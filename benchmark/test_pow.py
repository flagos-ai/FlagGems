import pytest
import torch

from . import base, consts


@pytest.mark.pow
def test_pow():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_
def test_pow_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
