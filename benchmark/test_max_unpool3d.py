import pytest
import torch

import flag_gems

from . import base, consts

# Benchmark shapes for max_unpool3d
MAX_UNPOOL3D_BENCH_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 16, 16, 16),
    (4, 4, 8, 8, 8),
]


class MaxUnpool3dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MAX_UNPOOL3D_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Generate pooled input and indices from max_pool3d
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
            output, indices = pool(x)
            # Output size should be the original input shape
            yield output, indices, 2, 2, 0, shape[
                2:
            ]  # kernel_size, stride, padding, output_size(D,H,W)


@pytest.mark.max_unpool3d
def test_max_unpool3d():
    def max_unpool3d_ref(input, indices, kernel_size, stride, padding, output_size):
        return torch.nn.functional.max_unpool3d(
            input,
            indices,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )

    bench = MaxUnpool3dBenchmark(
        op_name="max_unpool3d",
        torch_op=max_unpool3d_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_unpool3d)
    bench.run()
