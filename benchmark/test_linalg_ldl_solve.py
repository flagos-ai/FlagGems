import pytest
import torch

import flag_gems

from . import base

# LDL Solve benchmark
LDL_SOLVE_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (32, 7),
]

fp64_is_supported = flag_gems.runtime.device.support_fp64

# CUDA ldl_factor_ex used to build LD supports float32/float64/complex64/complex128 here.
LDL_SOLVE_DTYPES = [torch.float32, torch.float64, torch.complex64]
if fp64_is_supported:
    LDL_SOLVE_DTYPES.append(torch.complex128)


def _make_ldl_inputs(n, k, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.mT + torch.eye(n, dtype=dtype, device=device) * n
    LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    B = torch.randn(n, k, dtype=dtype, device=device)
    return LD, pivots, B


class LinalgLdlSolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_SOLVE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n, k = shape
            LD, pivots, B = _make_ldl_inputs(n, k, cur_dtype, self.device)
            yield LD, pivots, B


@pytest.mark.linalg_ldl_solve
def test_linalg_ldl_solve():
    bench = LinalgLdlSolveBenchmark(
        op_name="linalg_ldl_solve",
        torch_op=torch.linalg.ldl_solve,
        dtypes=LDL_SOLVE_DTYPES,
    )
    bench.run()
