import pytest
import torch

import flag_gems

from . import base

# Square matrix shapes for condition number computation (requires SVD/inverse)
LINALG_COND_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]


class LinalgCondBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_COND_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield A, None


@pytest.mark.linalg_cond
def test_linalg_cond():
    bench = LinalgCondBenchmark(
        op_name="linalg_cond",
        torch_op=torch.linalg.cond,
        # Only float32 supported: linalg.cond uses SVD which has limited dtype support on CUDA
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.linalg_cond)
    bench.run()
