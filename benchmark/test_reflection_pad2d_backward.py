from typing import Generator

import pytest
import torch

from . import base, consts, utils


class ReflectionPad2dBackwardBenchmark(base.Benchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape, padding in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            pad_h, pad_w = padding
            padding = (pad_w, pad_w, pad_h, pad_h)
            output_shape = (
                *shape[:-2],
                shape[-2] + padding[0] + padding[1],
                shape[-1] + padding[2] + padding[3],
            )
            grad_output = utils.generate_tensor_input(output_shape, dtype, self.device)
            yield grad_output, input, padding

    def get_bw_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape, padding in self.shapes:
            input = utils.generate_tensor_input(shape, dtype, self.device)
            pad_h, pad_w = padding
            padding = (pad_w, pad_w, pad_h, pad_h)
            output_shape = (
                *shape[:-2],
                shape[-2] + padding[0] + padding[1],
                shape[-1] + padding[2] + padding[3],
            )
            grad_output = utils.generate_tensor_input(output_shape, dtype, self.device)
            yield grad_output, input, padding


@pytest.mark.reflection_pad2d_backward
def test_reflection_pad2d_backward():
    bench = ReflectionPad2dBackwardBenchmark(
        op_name="reflection_pad2d_backward",
        torch_op=torch.ops.aten.reflection_pad2d_backward,
        dtypes=consts.FLOAT_DTYPES,
        shapes=[
            ((64, 64), (1, 1)),
            ((128, 128), (2, 2)),
            ((256, 256), (3, 3)),
            ((512, 512), (4, 4)),
        ],
    )
    bench.run()
