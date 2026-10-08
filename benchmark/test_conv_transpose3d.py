from typing import Generator

import pytest
import torch

from . import base, consts


class ConvTranspose3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (4, 16, 8, 8, 8, 32, 3, 1, 0, 1),
            (2, 32, 8, 16, 16, 16, 3, 2, 1, 1),
            (2, 16, 12, 12, 12, 32, 2, 1, 0, 1),
            (1, 32, 16, 16, 16, 16, 3, 2, 1, 2),
            (2, 24, 10, 10, 10, 24, 3, 1, 1, 3),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def conv_transpose3d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_d,
        input_h,
        input_w,
        out_c,
        kernel,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_d, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel, kernel, kernel)
    inp = base.generate_tensor_input(input_shape, dtype, device)
    weight = base.generate_tensor_input(weight_shape, dtype, device)

    yield (inp, weight, None, stride, padding, 0, groups)


@pytest.mark.conv_transpose3d
def test_conv_transpose3d():
    bench = ConvTranspose3dBenchmark(
        input_fn=conv_transpose3d_input_fn,
        op_name="conv_transpose3d",
        torch_op=torch.nn.functional.conv_transpose3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
