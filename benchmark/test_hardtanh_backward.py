import pytest
import torch

from . import base, consts


class HardtanhBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return None


def hardtanh_backward_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    min_val = -1.0
    max_val = 1.0
    yield grad_output, inp, min_val, max_val


@pytest.mark.hardtanh_backward
def test_hardtanh_backward():
    bench = HardtanhBackwardBenchmark(
        input_fn=hardtanh_backward_input_fn,
        op_name="hardtanh_backward",
        torch_op=torch.ops.aten.hardtanh_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
