from typing import Generator

import pytest
import torch

from . import base, consts, utils


def cudnn_convolution_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (
        inp,
        weight,
        [padding, padding],
        [stride, stride],
        [1, 1],
        groups,
        False,
        False,
        False,
    )


class CudnnConv2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    bench = CudnnConv2dBenchmark(
        input_fn=cudnn_convolution_input_fn,
        op_name="cudnn_convolution",
        torch_op=torch.ops.aten.cudnn_convolution.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
