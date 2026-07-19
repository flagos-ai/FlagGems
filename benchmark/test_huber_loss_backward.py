"""Performance benchmark for ``aten::huber_loss_backward`` against the
PyTorch native implementation."""

import pytest
import torch

from . import base, consts


class HuberLossBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (262144,),
        (1024, 1024),
        (4096, 4096),
        (64, 512, 512),
    ]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            target = torch.randn(shape, device=self.device, dtype=dtype)
            # 0-D grad_output is the common autograd case (reduced loss).
            grad_output = torch.randn((), device=self.device, dtype=dtype)
            yield grad_output, inp, target, 1, 1.0


@pytest.mark.huber_loss_backward
def test_huber_loss_backward():
    bench = HuberLossBackwardBenchmark(
        op_name="huber_loss_backward",
        torch_op=torch.ops.aten.huber_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
