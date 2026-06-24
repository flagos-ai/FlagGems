import pytest
import torch

from . import base, consts

# Shapes for prelu_kernel_backward benchmark
PRELU_KERNEL_BACKWARD_SHAPES = [
    (16, 128, 64, 1280),  # Large 4D shape
    (1024, 1024),  # 2D
    (16, 7, 57, 32, 29),  # 5D
]


class PReluKernelBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PRELU_KERNEL_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            grad_output = torch.randn(shape, dtype=cur_dtype, device=self.device)
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            weight = torch.tensor([0.25], dtype=cur_dtype, device=self.device)
            yield grad_output, x, weight


@pytest.mark.prelu_kernel_backward
def test_prelu_kernel_backward():
    bench = PReluKernelBackwardBenchmark(
        op_name="prelu_kernel_backward",
        torch_op=torch.ops.aten._prelu_kernel_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
