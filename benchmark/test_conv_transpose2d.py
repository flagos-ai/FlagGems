from typing import Generator

import pytest
import torch

from . import base, consts, utils


def conv_transpose2d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel, kernel)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (inp, weight, None, stride, padding, 0, groups)


class ConvTranspose2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            # batch, c_in, h, w, c_out, kernel, stride, padding, groups
            (16, 32, 32, 32, 64, 3, 1, 1, 1),
            (8, 64, 64, 64, 64, 3, 2, 1, 1),
            (4, 64, 128, 128, 32, 4, 2, 1, 1),
            (4, 32, 64, 64, 32, 3, 1, 0, 4),  # grouped
            (2, 16, 256, 256, 16, 3, 1, 1, 1),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.conv_transpose2d
def test_conv_transpose2d():
    bench = ConvTranspose2dBenchmark(
        input_fn=conv_transpose2d_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
