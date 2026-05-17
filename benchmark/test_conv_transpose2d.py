import pytest
import torch

import flag_gems

from . import base


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (16, 32, 32, 32, 64, 3, 3, 2, 1, 1),
            (32, 64, 32, 32, 32, 3, 3, 1, 0, 1),
            (32, 64, 16, 16, 32, 3, 3, 2, 1, 1),
        ]


def _input_fn(shape, dtype, device):
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
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield (
        {
            "input": inp,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },
    )


@pytest.mark.conv_transpose2d
def test_perf_conv_transpose2d(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=[torch.float16, torch.float32],
    )
    bench.set_gems(flag_gems.conv_transpose2d)
    bench.run()
