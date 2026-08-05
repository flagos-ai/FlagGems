import pytest
import torch

from . import base, consts


class HardswishBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def hardswish_backward_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    yield grad_output, inp


@pytest.mark.hardswish_backward
def test_hardswish_backward():
    bench = HardswishBackwardBenchmark(
        input_fn=hardswish_backward_input_fn,
        op_name="hardswish_backward",
        torch_op=torch.ops.aten.hardswish_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
