import math
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def _has_native_ascend_kernel() -> bool:
    if flag_gems.vendor_name != "ascend":
        return True
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::log_sigmoid_backward", "PrivateUse1"
        )
    except (AttributeError, RuntimeError):
        return False


_HAS_NATIVE_ASCEND_KERNEL = _has_native_ascend_kernel()
_SHAPE_REPLACEMENTS = {
    (1024 * 1024 * 1024,): (32 * 1024 * 1024,),
    (1024, 1024, 1024): (128, 512, 512),
}
_MIN_PERFORMANCE_ELEMENTS = 1024 * 1024


def torch_log_sigmoid_backward(grad_output, inp, buffer):
    if _HAS_NATIVE_ASCEND_KERNEL:
        return torch.ops.aten.log_sigmoid_backward(grad_output, inp, buffer)

    # Some Ascend PyTorch builds do not provide this ATen C kernel. In that
    # case compare against the equivalent composition of native device ops.
    return grad_output * torch.sigmoid(-inp)


def torch_log_sigmoid_backward_out(grad_output, inp, buffer, *, grad_input):
    if _HAS_NATIVE_ASCEND_KERNEL:
        return torch.ops.aten.log_sigmoid_backward.grad_input(
            grad_output, inp, buffer, grad_input=grad_input
        )

    grad_input.copy_(grad_output * torch.sigmoid(-inp))
    return grad_input


class LogSigmoidBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Keep standard medium and large pointwise throughput shapes. Shapes
        # below 1M elements remain covered by functional tests. Replace the
        # billion-element defaults because four live tensors make them unstable
        # on shared devices.
        self.shapes = [_SHAPE_REPLACEMENTS.get(shape, shape) for shape in self.shapes]
        self.shapes = [
            shape
            for shape in self.shapes
            if math.prod(shape) >= _MIN_PERFORMANCE_ELEMENTS
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            grad_output = base.generate_tensor_input(shape, cur_dtype, self.device)
            buffer = torch.exp(-torch.abs(inp))
            yield grad_output, inp, buffer


class LogSigmoidBackwardOutBenchmark(LogSigmoidBackwardBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for grad_output, inp, buffer in super().get_input_iter(cur_dtype):
            grad_input = torch.empty_like(inp)
            yield grad_output, inp, buffer, {"grad_input": grad_input}


@pytest.mark.log_sigmoid_backward
def test_log_sigmoid_backward():
    bench = LogSigmoidBackwardBenchmark(
        op_name="log_sigmoid_backward",
        torch_op=torch_log_sigmoid_backward,
        gems_op=flag_gems.log_sigmoid_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.log_sigmoid_backward_out
def test_log_sigmoid_backward_out():
    bench = LogSigmoidBackwardOutBenchmark(
        op_name="log_sigmoid_backward_out",
        torch_op=torch_log_sigmoid_backward_out,
        gems_op=flag_gems.log_sigmoid_backward_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
