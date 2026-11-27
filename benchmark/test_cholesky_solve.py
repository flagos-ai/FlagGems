import pytest
import torch

import flag_gems

from . import base

VENDOR_NAME = flag_gems.vendor_name
IS_ASCEND = VENDOR_NAME == "ascend"
IS_KUNLUNXIN = VENDOR_NAME == "kunlunxin"
IS_THEAD = VENDOR_NAME == "thead"
IS_MTHREADS = VENDOR_NAME == "mthreads"
SUPPORT_FP64 = flag_gems.runtime.device.support_fp64

REAL_DTYPES = [torch.float32] + ([torch.float64] if SUPPORT_FP64 else [])
COMPLEX_DTYPES = [torch.complex64] + ([torch.complex128] if SUPPORT_FP64 else [])

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
        # MUSA cannot reliably build Hermitian complex inputs with the vendor
        # torch stack (device cholesky reports non-positive-definite), so build
        # the complex setup on CPU and move it to the device for the op.
        build_on_cpu = IS_MTHREADS and cur_dtype in COMPLEX_DTYPES
        setup_device = "cpu" if build_on_cpu else self.device
        for shape, upper in self.shapes:
            *batch_dims, n, nrhs = shape
            B_mat = torch.randn(*batch_dims, n, n, dtype=cur_dtype, device=setup_device)
            eye = torch.eye(n, dtype=cur_dtype, device=setup_device)
            for _ in batch_dims:
                eye = eye.unsqueeze(0)
            A = B_mat @ B_mat.mH + eye * 0.1
            if build_on_cpu and not torch._C.has_lapack:
                import numpy as np

                L = torch.from_numpy(np.linalg.cholesky(A.numpy()))
            else:
                L = torch.linalg.cholesky(A)
            factor = L.mH.contiguous() if upper else L
            rhs = torch.randn(
                *batch_dims, n, nrhs, dtype=cur_dtype, device=setup_device
            )
            if build_on_cpu:
                factor = factor.to(self.device)
                rhs = rhs.to(self.device)
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
    else:
        torch_op = torch.ops.aten.cholesky_solve
        gems_op = None
        dtypes = (
            REAL_DTYPES if (IS_THEAD or IS_KUNLUNXIN) else REAL_DTYPES + COMPLEX_DTYPES
        )

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
    else:
        torch_op = torch.cholesky_solve
        gems_op = None
        dtypes = (
            REAL_DTYPES if (IS_THEAD or IS_KUNLUNXIN) else REAL_DTYPES + COMPLEX_DTYPES
        )

    bench = CholeskySolveOutBenchmark(
        op_name="cholesky_solve_out",
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=dtypes,
    )
    bench.run()


# (B shape, upper): the last dim is the number of right-hand sides.
_CHOLESKY_SOLVE_BACKWARD_SHAPES = [
    ((32, 32), False),
    ((64, 16), False),
    ((64, 16), True),
    ((8, 128, 32), False),
]


def cholesky_solve_backward_torch_op(B, L, upper):
    # The VJP of ``cholesky_solve`` w.r.t. its right-hand side is A^-T grad,
    # and A is symmetric, so it is A^-1 * ones -- two triangular solves with
    # the factor and its transpose. Solving it explicitly keeps the reference
    # path inside cuBLAS: ATen's own ``cholesky_solve_backward`` dispatches a
    # *batched* ``cholesky_solve``, which lands in MAGMA's batched TRSM
    # (``magma_spotrs_batched``) and aborts the whole process under CUDA Graph
    # capture. That abort is a hard crash, not a catchable exception, so the
    # harness cannot fall back and every candidate reads as RUNTIME_ERROR no
    # matter how correct its kernel is. Verified numerically equal to
    # ``torch.autograd.grad`` in float64.
    ones = torch.ones_like(B)
    if upper:
        # A = U^T U, so A^-1 = U^-1 U^-T.
        y = torch.linalg.solve_triangular(L.transpose(-1, -2), ones, upper=False)
        return torch.linalg.solve_triangular(L, y, upper=True)
    # A = L L^T, so A^-1 = L^-T L^-1.
    y = torch.linalg.solve_triangular(L, ones, upper=False)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)


def cholesky_solve_backward_case_fn(case, dtype):
    yield base.BenchmarkCasePlan(
        shape={"B": list(case[0])},
        params={"upper": case[1]},
        builder_args=(case,),
    )


def cholesky_solve_backward_build_inputs_fn(plan, dtype, device):
    shape, upper = plan.builder_args[0]
    n = shape[-2]
    L = torch.randn((*shape[:-2], n, n), dtype=dtype, device=device)
    L = (L @ L.transpose(-1, -2)).tril()
    L.diagonal(dim1=-2, dim2=-1).add_(1.0)
    if upper:
        L = L.transpose(-1, -2).contiguous()
    B = torch.randn(shape, dtype=dtype, device=device)
    B = B.requires_grad_(True)
    return B, L, upper


class CholeskySolveBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _CHOLESKY_SOLVE_BACKWARD_SHAPES
        self.shape_desc = "(B shape, upper)"


@pytest.mark.cholesky_solve_backward
def test_cholesky_solve_backward():
    bench = CholeskySolveBackwardBenchmark(
        case_fn=cholesky_solve_backward_case_fn,
        build_inputs_fn=cholesky_solve_backward_build_inputs_fn,
        op_name="cholesky_solve_backward",
        torch_op=cholesky_solve_backward_torch_op,
        # CUDA's cholesky only supports fp32/fp64; fp16/bf16 raise
        # "cholesky_cuda_potrs not implemented", so they cannot be benchmarked.
        dtypes=[torch.float32],
    )
    bench.run()
