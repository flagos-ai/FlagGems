# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config

VENDOR = flag_gems.vendor_name

# On ascend the general KERNEL-mode do_bench_npu is unreliable; use the
# operator (end-to-end) timing mode (same as det/lu_factor/cholesky_solve).
if VENDOR == "ascend":
    Config.mode = consts.BenchMode.OPERATOR

SOLVE_TRI_SHAPES = [
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 256),
]

SOLVE_TRI_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64 and VENDOR != "ascend":
    SOLVE_TRI_DTYPES.append(torch.float64)


def _solve_tri_small_ops(A, B, upper=False, left=True, unitriangular=False, **kwargs):
    """Small-op combination baseline (block forward/backward substitution via
    torch.matmul + torch.linalg.inv), running on AI Core.

    On ascend the torch reference (aclnnTriangularSolve) runs on AI_CPU, which
    is not a meaningful comparison; this small-op combination is used as the
    benchmark baseline instead (same approach as det / lu_factor).  Its
    agreement with torch.linalg.solve_triangular is validated by
    test_baseline_matches_torch in tests/test_linalg_solve_triangular.py.
    """
    n = A.shape[-1]
    bs = 64
    X = B.clone()
    if not left:
        # X A = B  <=>  A^T X^T = B^T  (reduce to left-multiply, transpose back)
        return _solve_tri_small_ops(
            A.mT.contiguous(),
            B.mT.contiguous(),
            not upper,
            True,
            unitriangular,
        ).mT.contiguous()
    if upper:
        for i in range(n - 1, -1, -bs):
            i0 = max(0, i - bs + 1)
            i1 = i + 1
            Aii = A[..., i0:i1, i0:i1]
            rhs = B[..., i0:i1, :]
            if i1 < n:
                rhs = rhs - torch.matmul(A[..., i0:i1, i1:], X[..., i1:, :])
            X[..., i0:i1, :] = torch.matmul(torch.linalg.inv(Aii), rhs)
    else:
        for i in range(0, n, bs):
            i1 = min(i + bs, n)
            Aii = A[..., i:i1, i:i1]
            rhs = B[..., i:i1, :]
            if i > 0:
                rhs = rhs - torch.matmul(A[..., i:i1, :i], X[..., :i, :])
            X[..., i:i1, :] = torch.matmul(torch.linalg.inv(Aii), rhs)
    return X


def _torch_solve_tri(A, B, **kwargs):
    """Benchmark baseline: on NPU use the small-op combination (AI Core);
    elsewhere use the torch reference."""
    if A.device.type == "npu":
        return _solve_tri_small_ops(A, B, **kwargs)
    return torch.linalg.solve_triangular(A, B, **kwargs)


def _make_triangular_input(n, k, dtype, device, upper, unitriangular):
    """Generate a well-conditioned triangular matrix: A = I + 0.1 * tri(randn)"""
    A = torch.randn(n, n, dtype=dtype, device=device)
    off_diag = 0.1
    if upper:
        A = A.triu(diagonal=1)
    else:
        A = A.tril(diagonal=-1)
    A.mul_(off_diag)
    A.add_(torch.eye(n, dtype=dtype, device=device))
    if unitriangular:
        A.diagonal().fill_(1.0)
    B = torch.randn(n, k, dtype=dtype, device=device)
    return A, B


class SolveTriBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOLVE_TRI_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, k in self.shapes:
            for upper in (False, True):
                A, B = _make_triangular_input(
                    n, k, cur_dtype, self.device, upper, False
                )
                yield A, B, {"upper": upper}


@pytest.mark.linalg_solve_triangular
def test_linalg_solve_triangular():
    bench = SolveTriBenchmark(
        op_name="linalg_solve_triangular",
        torch_op=_torch_solve_tri,
        gems_op=flag_gems.linalg_solve_triangular,
        dtypes=SOLVE_TRI_DTYPES,
    )
    bench.run()


class SolveTriOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOLVE_TRI_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, k in self.shapes:
            for upper in (False, True):
                A, B = _make_triangular_input(
                    n, k, cur_dtype, self.device, upper, False
                )
                out = torch.empty_like(B)
                yield A, B, {"upper": upper, "out": out}


@pytest.mark.linalg_solve_triangular_out
def test_linalg_solve_triangular_out():
    bench = SolveTriOutBenchmark(
        op_name="linalg_solve_triangular_out",
        torch_op=_torch_solve_tri,
        gems_op=flag_gems.linalg_solve_triangular_out,
        dtypes=SOLVE_TRI_DTYPES,
    )
    bench.run()
