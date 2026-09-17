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

from . import base

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
if flag_gems.runtime.device.support_fp64:
    SOLVE_TRI_DTYPES.append(torch.float64)


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
        torch_op=torch.linalg.solve_triangular,
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
        torch_op=torch.linalg.solve_triangular,
        gems_op=flag_gems.linalg_solve_triangular_out,
        dtypes=SOLVE_TRI_DTYPES,
    )
    bench.run()
