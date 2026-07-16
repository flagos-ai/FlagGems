import pytest
import torch

from flag_gems import linalg_slogdet

from . import base

# Use linalg-specific square-matrix shapes: one batched small case covers
# the (*, n, n) interface, and 4x4 through 32x32 covers the small/medium
# matrices targeted by this single-program LU implementation.
SLOGDET_SHAPES = [
    (2, 3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark.linalg_slogdet
def test_linalg_slogdet():
    bench = SlogdetBenchmark(
        op_name="linalg_slogdet",
        torch_op=torch.linalg.slogdet,
        # linalg.slogdet generated kernel only supports float32 on CUDA.
        dtypes=[torch.float32],
    )
    bench.set_gems(linalg_slogdet)
    bench.run()
