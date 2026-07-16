import pytest
import torch

from . import consts
from .test_bitwise_left_shift import BitwiseLeftShiftBenchmark


@pytest.mark.bitwise_left_shift_
def test_bitwise_left_shift_():
    bench = BitwiseLeftShiftBenchmark(
        op_name="bitwise_left_shift_",
        torch_op=torch.Tensor.bitwise_left_shift_,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )

    bench.run()
