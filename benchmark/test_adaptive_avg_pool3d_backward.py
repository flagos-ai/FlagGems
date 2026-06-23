import pytest
import torch

from . import base, consts

# Shapes for adaptive_avg_pool3d backward benchmark
ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 32, 32, 32),
    (4, 8, 64, 64, 64),
]


class AdaptiveAvgPool3DBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES
        self.output_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]

    def get_input_iter(self, cur_dtype):
        for shape, output_size in zip(self.shapes, self.output_sizes):
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Compute forward to get output shape
            out = torch.nn.functional.adaptive_avg_pool3d(x, output_size)
            grad = torch.ones_like(out)
            yield grad, x


@pytest.mark.adaptive_avg_pool3d_backward
def test_adaptive_avg_pool3d_backward():
    bench = AdaptiveAvgPool3DBackwardBenchmark(
        op_name="adaptive_avg_pool3d_backward",
        torch_op=torch.ops.aten._adaptive_avg_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
