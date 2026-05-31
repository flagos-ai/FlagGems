from typing import Generator

import pytest
import torch

from . import base, consts, utils


class UpsampleNearestExact2DBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device)
            yield input, output_size

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device, requires_grad=True)
            grad_output = utils.generate_tensor_input(output_size, dtype, self.device)
            yield grad_output, input_size, output_size


@pytest.mark.upsample_nearest_exact2d
def test_upsample_nearest_exact2d():
    bench = UpsampleNearestExact2DBenchmark(
        op_name="_upsample_nearest_exact2d",
        torch_op=lambda x, sz: torch.ops.aten._upsample_nearest_exact2d(x, sz),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[((1, 3, 16, 16), (32, 32)), ((2, 16, 32, 32), (64, 48))],
    )
    bench.run()


@pytest.mark.upsample_nearest_exact2d_backward
def test_upsample_nearest_exact2d_backward():
    bench = UpsampleNearestExact2DBenchmark(
        op_name="_upsample_nearest_exact2d_backward",
        torch_op=lambda go, sz, isz: torch.ops.aten._upsample_nearest_exact2d_backward(go, isz, sz),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[((1, 3, 16, 16), (32, 32)), ((2, 16, 32, 32), (64, 48))],
    )
    bench.run()
