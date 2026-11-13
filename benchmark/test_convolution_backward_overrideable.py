import pytest
import torch

import flag_gems

from . import base, consts


class ConvBackwardOverrideableBenchmark(base.GenericBenchmark):
    # shape = (batch, in_c, in_h, in_w, out_c, kh, kw, stride, padding, groups)
    DEFAULT_SHAPES = [
        (32, 64, 128, 128, 32, 3, 3, 1, 1, 1),
        (32, 64, 64, 64, 16, 5, 5, 2, 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 1, 0, 1),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.DEFAULT_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


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
    input = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)
    out = torch.nn.functional.conv2d(
        input, weight, bias=None, stride=stride, padding=padding, groups=groups
    )
    grad_output = torch.randn_like(out)

    yield {
        "grad_output": grad_output,
        "input": input,
        "weight": weight,
        "stride": [stride, stride],
        "padding": [padding, padding],
        "dilation": [1, 1],
        "groups": groups,
    },


def _torch_ref(grad_output, input, weight, stride, padding, dilation, groups):
    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        [weight.shape[0]],
        stride,
        padding,
        dilation,
        False,
        [0, 0],
        groups,
        [True, True, True],
    )


def _gems_op(grad_output, input, weight, stride, padding, dilation, groups):
    return flag_gems.convolution_backward_overrideable(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        False,
        [0, 0],
        groups,
        [True, True, True],
    )


@pytest.mark.convolution_backward_overrideable
def test_convolution_backward_overrideable(monkeypatch):
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvBackwardOverrideableBenchmark(
        input_fn=_input_fn,
        op_name="convolution_backward_overrideable",
        torch_op=_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(_gems_op)
    bench.run()


@pytest.mark.convolution_backward_overrideable_out
def test_convolution_backward_overrideable_out(monkeypatch):
    torch.backends.cudnn.allow_tf32 = False

    def _gems_out_op(grad_output, input, weight, stride, padding, dilation, groups):
        out0 = torch.empty_like(input)
        out1 = torch.empty_like(weight)
        out2 = torch.empty(weight.shape[0], dtype=input.dtype, device=input.device)
        return flag_gems.convolution_backward_overrideable_out(
            grad_output,
            input,
            weight,
            stride,
            padding,
            dilation,
            False,
            [0, 0],
            groups,
            [True, True, True],
            out0,
            out1,
            out2,
        )

    bench = ConvBackwardOverrideableBenchmark(
        input_fn=_input_fn,
        op_name="convolution_backward_overrideable_out",
        torch_op=_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(_gems_out_op)
    bench.run()
