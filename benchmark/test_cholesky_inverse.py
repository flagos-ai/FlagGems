import pytest
import torch

from . import base

# Square matrices from 2x2 to 256x256 covering small to medium-large use cases
CHOLESKY_INVERSE_SHAPES = [
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]


class CholeskyInverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_INVERSE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[-1]
            # Create positive-definite matrix and get its Cholesky factor
            B = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = (
                B @ B.transpose(-2, -1)
                + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            )
            L = torch.linalg.cholesky(A)
            yield (L,)


@pytest.mark.cholesky_inverse
def test_cholesky_inverse():
    bench = CholeskyInverseBenchmark(
        op_name="cholesky_inverse",
        torch_op=torch.cholesky_inverse,
        # cholesky_inverse only supports float32/float64
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
