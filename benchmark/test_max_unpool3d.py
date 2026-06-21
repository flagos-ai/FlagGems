import pytest
import torch

from . import base, consts


class MaxUnpool3dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # 3D shapes (N, C, D, H, W) where D/H/W are divisible by 2
        # to produce valid pooled sizes for kernel_size=2, stride=2
        self.shapes = [
            (1, 1, 4, 4, 4),
            (1, 1, 8, 8, 8),
            (2, 4, 4, 4, 4),
            (2, 8, 8, 8, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            N, C, D, H, W = shape
            # Create input and do max pooling to get pooled output and indices
            input_orig = torch.randn(shape, dtype=cur_dtype, device=self.device)
            pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
            pooled, indices = pool(input_orig)
            yield pooled, indices, 2  # kernel_size=2


@pytest.mark.max_unpool3d
def test_max_unpool3d_perf():
    bench = MaxUnpool3dBenchmark(
        op_name="max_unpool3d",
        torch_op=torch.nn.functional.max_unpool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
