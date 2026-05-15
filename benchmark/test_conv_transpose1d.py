from typing import Generator

import pytest
import torch

from . import base, consts, utils


def conv_transpose1d_input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_l,
        out_c,
        kernel,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_l)
    weight_shape = (input_c, out_c // groups, kernel)
    inp = utils.generate_tensor_input(input_shape, dtype, device)
    weight = utils.generate_tensor_input(weight_shape, dtype, device)

    yield (inp, weight, None, stride, padding, 0, groups)


class ConvTranspose1dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes = [
            (32, 64, 128, 64, 3, 1, 0, 1),
            (64, 48, 256, 128, 5, 2, 2, 1),
            (16, 24, 512, 96, 7, 1, 3, 1),
            (8, 16, 1024, 32, 3, 2, 1, 2),
            (4, 8, 2048, 16, 5, 1, 2, 1),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.conv_transpose1d
def test_conv_transpose1d():
    bench = ConvTranspose1dBenchmark(
        input_fn=conv_transpose1d_input_fn,
        op_name="conv_transpose1d",
        torch_op=torch.nn.functional.conv_transpose1d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
