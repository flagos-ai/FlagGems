import pytest
import torch

import flag_gems

from . import base

VENDOR_NAME = flag_gems.vendor_name
IS_ASCEND = VENDOR_NAME == "ascend"
IS_THEAD = VENDOR_NAME == "thead"

if IS_ASCEND:
    from flag_gems.runtime.backend._ascend.ops.cholesky_solve import (
        cholesky_solve,
        cholesky_solve_out,
    )

# Two-dimensional entries are single systems; longer entries benchmark batched
# solves. Keep one shape list and derive both factor orientations from it so
# every lower/upper latency comparison has an exact counterpart.
CHOLESKY_SOLVE_SHAPES = [
    # Single RHS: exposes the low-parallelism triangular-solve path.
    (8, 1),
    (16, 1),
    (32, 1),
    # Real small-gather single-RHS range (BLOCK_N=64).
    (33, 1),
    (48, 1),
    (63, 1),
    (64, 1),
    (128, 1),
    (256, 1),
    # Small-N small-RHS fused path coverage.
    (16, 2),
    (16, 4),
    (32, 4),
    # RHS sweep around BLOCK_RHS boundaries and tail cases.
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 31),
    (64, 32),
    (64, 33),
    (64, 64),
    (64, 128),
    # Throughput-oriented larger systems.
    (128, 16),
    (128, 64),
    (256, 16),
    (256, 128),
    # Batched systems: important for occupancy with one batch per program tile.
    (16, 16, 1),
    (64, 16, 1),
    (256, 16, 1),
    (16, 16, 4),
    (16, 32, 4),
    (16, 32, 8),
    (32, 64, 16),
    (8, 128, 16),
]

# Case format: ((*batch_dims, N, nrhs), upper). Group lower then upper while
# preserving identical shape order to make benchmark tables easy to compare.
CHOLESKY_SOLVE_CASES = [
    (shape, upper) for upper in (False, True) for shape in CHOLESKY_SOLVE_SHAPES
]

# Keep the out benchmark intentionally small. These cases cover the main
# dispatch paths without duplicating the complete functional benchmark matrix:
# small-N, blocked single-RHS, an RHS tile tail, large multi-RHS, and batching.
CHOLESKY_SOLVE_OUT_CASES = [
    ((16, 4), False),
    ((64, 1), True),
    ((64, 33), False),
    ((256, 128), True),
    ((8, 128, 16), False),
]


class CholeskySolveBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_CASES
        self.shape_desc = "((*batch, N, nrhs), upper)"

    def get_input_iter(self, cur_dtype):
        for shape, upper in self.shapes:
            *batch_dims, n, nrhs = shape
            B_mat = torch.randn(*batch_dims, n, n, dtype=cur_dtype, device=self.device)
            eye = torch.eye(n, dtype=cur_dtype, device=self.device)
            for _ in batch_dims:
                eye = eye.unsqueeze(0)
            A = B_mat @ B_mat.mH + eye * 0.1
            L = torch.linalg.cholesky(A)
            factor = L.mH.contiguous() if upper else L
            rhs = torch.randn(*batch_dims, n, nrhs, dtype=cur_dtype, device=self.device)
            yield (rhs, factor, upper)


class CholeskySolveOutBenchmark(CholeskySolveBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CHOLESKY_SOLVE_OUT_CASES
        self.shape_desc = "((*batch, N, nrhs), upper, out)"

    def get_input_iter(self, cur_dtype):
        for rhs, factor, upper in super().get_input_iter(cur_dtype):
            yield rhs, factor, upper, {"out": torch.empty_like(rhs)}


def _composed_cholesky_solve(rhs, factor, upper=False):
    """Reference composed from two native triangular solves on Ascend."""
    if upper:
        y = torch.linalg.solve_triangular(factor.mH, rhs, upper=False)
        return torch.linalg.solve_triangular(factor, y, upper=True)

    y = torch.linalg.solve_triangular(factor, rhs, upper=False)
    return torch.linalg.solve_triangular(factor.mH, y, upper=True)


def _composed_cholesky_solve_out(rhs, factor, upper=False, *, out):
    out.copy_(_composed_cholesky_solve(rhs, factor, upper=upper))
    return out


@pytest.mark.cholesky_solve
def test_cholesky_solve():
    if IS_ASCEND:
        # torch.cholesky_solve has no native NPU implementation. Compare the
        # fused Ascend kernel against two native solve_triangular operators.
        torch_op = _composed_cholesky_solve
        gems_op = cholesky_solve
        dtypes = [torch.float32]
    elif IS_THEAD:
        # Thead torch.ops.aten.cholesky_solve do not support complex dtype
        torch_op = torch.ops.aten.cholesky_solve
        gems_op = None
        dtypes = [torch.float32, torch.float64]
    else:
        torch_op = torch.ops.aten.cholesky_solve
        gems_op = None
        dtypes = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    bench = CholeskySolveBenchmark(
        op_name="cholesky_solve",
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=dtypes,
    )
    bench.run()


@pytest.mark.cholesky_solve_out
def test_cholesky_solve_out():
    if IS_ASCEND:
        torch_op = _composed_cholesky_solve_out
        gems_op = cholesky_solve_out
        dtypes = [torch.float32]
    elif IS_THEAD:
        # Thead torch.cholesky_solve does not support complex dtype.
        torch_op = torch.cholesky_solve
        gems_op = None
        dtypes = [torch.float32, torch.float64]
    else:
        torch_op = torch.cholesky_solve
        gems_op = None
        dtypes = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    bench = CholeskySolveOutBenchmark(
        op_name="cholesky_solve_out",
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=dtypes,
    )
    bench.run()
