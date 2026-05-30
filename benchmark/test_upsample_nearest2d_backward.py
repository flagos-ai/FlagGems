from typing import Generator

import pytest
import torch

from . import base, consts, utils


class UpsampleNearest2DBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device, requires_grad=True)
            yield input, output_size

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device, requires_grad=True)
            grad_output = utils.generate_tensor_input(output_size, dtype, self.device)
            yield grad_output, input_size, output_size


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleNearest2DBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        torch_op=lambda x, sz: torch.ops.aten.upsample_nearest2d(x, sz).backward(torch.ones_like(torch.ops.aten.upsample_nearest2d(x, sz))),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[((1, 3, 16, 16), (32, 32)), ((2, 16, 32, 32), (64, 64))],
    )
    bench.run()
