import pytest
import torch

import flag_gems

from . import base, consts


class ConvTbcBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype):
        # (time, batch, in_channels, out_channels, kernel_width, pad)
        shapes = [
            (64, 16, 64, 64, 3, 1),
            (128, 32, 128, 128, 3, 1),
            (256, 16, 256, 256, 5, 2),
            (512, 8, 128, 128, 3, 1),
            (1024, 4, 64, 64, 7, 3),
        ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def conv_tbc_backward_input_fn(shape, dtype, device):
    ilen, batch, in_c, out_c, kw, pad = shape
    olen = ilen - kw + 1 + 2 * pad
    inp = torch.randn((ilen, batch, in_c), dtype=dtype, device=device)
    weight = torch.randn((kw, in_c, out_c), dtype=dtype, device=device)
    bias = torch.randn((out_c,), dtype=dtype, device=device)
    grad_output = torch.randn((olen, batch, out_c), dtype=dtype, device=device)
    yield grad_output, inp, weight, bias, pad


@pytest.mark.conv_tbc_backward
def test_conv_tbc_backward():
    bench = ConvTbcBackwardBenchmark(
        input_fn=conv_tbc_backward_input_fn,
        op_name="conv_tbc_backward",
        torch_op=torch.ops.aten.conv_tbc_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_tbc_backward)
    bench.run()
