import pytest
import torch

from . import base

# Benchmark shapes: (batch, n, k) - focus on batched small/medium matrices
# where fused Triton kernel amortizes launch overhead vs cuSOLVER per-element calls
SOLVE_EX_SHAPES = [
    (1, 4, 1),
    (1, 8, 1),
    (1, 16, 1),
    (4, 4, 4),
    (4, 8, 4),
    (4, 16, 4),
    (8, 8, 1),
    (8, 16, 1),
    (16, 8, 4),
    (16, 16, 4),
    (32, 8, 1),
]

# LU decomposition requires high precision; float16/bfloat16 lack mantissa bits
SOLVE_EX_DTYPES = [torch.float32]


def _solve_ex_input_fn(batch, n, k, dtype, device):
    """Create well-conditioned batched A and random B."""
    A = torch.randn(batch, n, n, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch, n, n)
    A = A @ A.mT + eye * n
    B = torch.randn(batch, n, k, dtype=dtype, device=device)
    return A, B


class LinalgSolveExBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOLVE_EX_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            batch, n, k = shape
            A, B = _solve_ex_input_fn(batch, n, k, cur_dtype, self.device)
            yield A, B


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex():
    bench = LinalgSolveExBenchmark(
        op_name="linalg_solve_ex",
        torch_op=torch.linalg.solve_ex,
        dtypes=SOLVE_EX_DTYPES,
    )
    bench.run()
