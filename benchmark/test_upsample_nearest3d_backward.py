from typing import Generator

import pytest
import torch

from . import base, consts, utils


class UpsampleNearest3DBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device, requires_grad=True)
            yield input, output_size

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for (input_size, output_size) in self.shapes:
            input = utils.generate_tensor_input(input_size, dtype, self.device, requires_grad=True)
            grad_output = utils.generate_tensor_input(output_size, dtype, self.device)
            yield grad_output, input_size, output_size


@pytest.mark.upsample_nearest3d_backward
def test_upsample_nearest3d_backward():
    bench = UpsampleNearest3DBackwardBenchmark(
        op_name="upsample_nearest3d_backward",
        torch_op=lambda x, sz: torch.ops.aten.upsample_nearest3d(x, sz).backward(torch.ones_like(torch.ops.aten.upsample_nearest3d(x, sz))),
        dtypes=consts.FLOAT_DTYPES,
        shapes=[((1, 3, 8, 8, 8), (16, 16, 16)), ((2, 16, 16, 16, 16), (32, 32, 32))],
    )
    bench.run()
