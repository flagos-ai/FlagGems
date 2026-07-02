import pytest
import torch

from . import base

# LDL Solve benchmark
LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]

# CUDA ldl_factor_ex used to build LD supports only float32 and float64 here.
LDL_SOLVE_DTYPES = [torch.float32, torch.float64]


class LinalgLdlSolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_SOLVE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n, k = shape
            # Create a symmetric positive definite matrix
            A = torch.randn(n, n, dtype=cur_dtype, device=self.device)
            A = A @ A.mT + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            # Compute LDL factorization
            LD, pivots, info = torch.linalg.ldl_factor_ex(A)
            # Right-hand side
            B = torch.randn(n, k, dtype=cur_dtype, device=self.device)
            yield LD, pivots, B


@pytest.mark.linalg_ldl_solve
def test_linalg_ldl_solve():
    bench = LinalgLdlSolveBenchmark(
        op_name="linalg_ldl_solve",
        torch_op=torch.linalg.ldl_solve,
        dtypes=LDL_SOLVE_DTYPES,
    )
    bench.run()
