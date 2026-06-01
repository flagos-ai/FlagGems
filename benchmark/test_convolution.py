import pytest
import torch

from . import base, consts


class ConvolutionBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 128, 128, 32, 3, 3, 1, 1, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
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
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    stride_list = [stride, stride]
    padding_list = [padding, padding]
    dilation_list = [1, 1]
    output_padding_list = [0, 0]

    yield inp, {
        "weight": weight,
        "bias": None,
        "stride": stride_list,
        "padding": padding_list,
        "dilation": dilation_list,
        "transposed": False,
        "output_padding": output_padding_list,
        "groups": groups,
        "benchmark": False,
        "deterministic": False,
        "cudnn_enabled": True,
        "allow_tf32": True,
    }


@pytest.mark.convolution
def test_convolution(monkeypatch):
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvolutionBenchmark(
        input_fn=_input_fn,
        op_name="convolution",
        torch_op=torch._convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
