import pytest
import torch

import flag_gems

from . import base

# LU Solve benchmark shapes: (n, k)
LU_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]

# LU solve supports float32/float64
fp64_is_supported = flag_gems.runtime.device.support_fp64

LU_SOLVE_DTYPES = [torch.float32]
if fp64_is_supported:
    LU_SOLVE_DTYPES.append(torch.float64)


def _make_lu_inputs(n, k, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    # Make A well-conditioned by adding n * I
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    LU, pivots = torch.linalg.lu_factor(A)
    B = torch.randn(n, k, dtype=dtype, device=device)
    return LU, pivots, B


class LinalgLuSolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LU_SOLVE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n, k = shape
            LU, pivots, B = _make_lu_inputs(n, k, cur_dtype, self.device)
            yield LU, pivots, B


@pytest.mark.linalg_lu_solve
def test_linalg_lu_solve():
    bench = LinalgLuSolveBenchmark(
        op_name="linalg_lu_solve",
        torch_op=torch.linalg.lu_solve,
        dtypes=LU_SOLVE_DTYPES,
    )
    bench.run()
