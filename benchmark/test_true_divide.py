import pytest
import torch

from . import base, consts

SCALAR = 0.3


@pytest.mark.div_tensor
def test_true_divide():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.true_divide,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_scalar
def test_true_divide_scalar():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_scalar",
        torch_op=lambda x, _: torch.true_divide(x, SCALAR),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_scalar_
def test_true_divide_inplace_scalar():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_scalar_",
        torch_op=lambda x, _: x.true_divide_(SCALAR),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
