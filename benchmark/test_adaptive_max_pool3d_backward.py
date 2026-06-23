import pytest
import torch

from . import base, consts

# Adaptive pooling benchmark shapes covering common 3D input sizes
ADAPTIVE_MAX_POOL3D_SHAPES = [
    (1, 1, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 32, 32, 32),
    (2, 8, 8, 8, 8),
]


class AdaptiveMaxPool3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ADAPTIVE_MAX_POOL3D_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            # Compute forward pass to get indices
            output_size = (shape[2] // 2, shape[3] // 2, shape[4] // 2)
            _, indices = torch.nn.functional.adaptive_max_pool3d(
                x, output_size=output_size, return_indices=True
            )
            grad_output = torch.ones_like(_)
            yield grad_output, x, indices


@pytest.mark.adaptive_max_pool3d_backward
def test_adaptive_max_pool3d_backward():
    bench = AdaptiveMaxPool3dBackwardBenchmark(
        op_name="adaptive_max_pool3d_backward",
        torch_op=torch.ops.aten.adaptive_max_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
