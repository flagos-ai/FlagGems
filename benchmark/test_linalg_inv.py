import pytest
import torch

from . import base

# Only 2x2 and 3x3 matrices supported by Triton kernel
LINALG_INV_SHAPES = [(2, 2), (3, 3)]


class LinalgInvBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LINALG_INV_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[0]
            # Create invertible matrix by adding identity
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = A + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            yield A,


@pytest.mark.linalg_inv
def test_linalg_inv():
    bench = LinalgInvBenchmark(
        op_name="linalg_inv",
        torch_op=lambda a: torch.linalg.inv(a),
        # torch.linalg.inv only supports float32 and float64 on CUDA, not float16/bfloat16
        dtypes=[torch.float32],
    )
    bench.run()
