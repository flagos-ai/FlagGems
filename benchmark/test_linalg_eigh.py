import pytest
import torch

from . import base

# Shapes for linalg_eigh benchmark (square matrices)
EIG_BENCHMARK_SHAPES = [
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
]


class LinalgEighBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EIG_BENCHMARK_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (A + A.transpose(-2, -1)) / 2
            yield A,


@pytest.mark.linalg_eigh
def test_linalg_eigh():
    bench = LinalgEighBenchmark(
        op_name="linalg_eigh",
        torch_op=torch.linalg.eigh,
        # linalg_eigh only supports float32/float64 on GPU
        dtypes=[torch.float32],
    )
    bench.run()
