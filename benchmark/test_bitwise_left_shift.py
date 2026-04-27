from typing import Generator

import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


class BitwiseLeftShiftBenchmark(base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp1 = utils.generate_tensor_input(shape, dtype, self.device)
            shift_amount = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(
                self.device
            )
            yield inp1, shift_amount


@pytest.mark.bitwise_left_shift
def test_bitwise_left_shift():
    bench = BitwiseLeftShiftBenchmark(
        op_name="bitwise_left_shift",
        torch_op=torch.bitwise_left_shift,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
