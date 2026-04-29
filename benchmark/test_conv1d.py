import pytest
import torch

import flag_gems

from . import base


class Conv1DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 512, 64, 3, 1, 0, 1),
            (64, 48, 1024, 128, 5, 2, 2, 1),
            (16, 24, 2048, 96, 7, 1, 3, 2),
            (8, 8, 8192, 16, 11, 4, 5, 1),
            (4, 4, 16384, 4, 15, 2, 7, 1),
            (32, 64, 512, 64, 3, 1, "valid", 1),
            (64, 48, 1024, 128, 5, 2, "valid", 1),
            (16, 24, 2048, 96, 7, 1, "same", 2),
            (8, 8, 8192, 16, 11, 1, "same", 1),
        ]


def conv1d_input_fn(shape, dtype, device):
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
    weight_shape = (out_c, input_c // groups, kernel)
    input = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield {
        "input": input,
        "weight": weight,
        "bias": None,
        "groups": groups,
        "stride": stride,
        "padding": padding,
    },


@pytest.mark.conv1d
def test_conv1d():
    torch.backends.cudnn.allow_tf32 = False
    bench = Conv1DBenchmark(
        input_fn=conv1d_input_fn,
        op_name="conv1d",
        torch_op=torch.nn.functional.conv1d,
        dtypes=[
            torch.float16,
            torch.float32,
        ],  # Exclude bfloat16 due to cuDNN limitations
    )
    bench.set_gems(flag_gems.conv1d)
    bench.run()
