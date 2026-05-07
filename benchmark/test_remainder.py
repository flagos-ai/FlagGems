import pytest
import torch

from . import base, consts


@pytest.mark.remainder_tensor
def test_remainder_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder",
        torch_op=torch.remainder,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.remainder_tensor_
def test_remainder_tensor_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
