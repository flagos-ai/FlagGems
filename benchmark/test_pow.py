import pytest
import torch

from . import base, consts


@pytest.mark.pow_tensor_tensor
def test_pow_tensor_tensor():
    bench = base.ScalarBinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor",
        torch_op=torch.pow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.pow_tensor_tensor_
def test_pow_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="pow_tensor_tensor_",
        torch_op=lambda a, b: a.pow_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
