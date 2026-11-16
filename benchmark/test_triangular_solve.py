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

import itertools

import pytest
import torch

import flag_gems

from . import base
from .consts import BenchmarkCasePlan

DTYPES = [
    pytest.param(torch.float32, id="float32"),
    pytest.param(
        torch.float64,
        id="float64",
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_fp64,
            reason="backend does not support float64",
        ),
    ),
]
# Both variants benchmark exactly these 12 systems crossed with every flag
# combination. Native availability never changes this declared workload.
# Format: (A batch, B batch, matrix order, number of RHS columns).
SYSTEMS = [
    ((), (), 8, 1),
    ((), (), 8, 8),
    ((), (), 32, 1),
    ((), (), 32, 64),
    ((), (), 128, 1),
    ((), (), 128, 8),
    ((), (), 512, 8),
    ((), (), 512, 64),
    ((), (), 1024, 1),
    ((), (), 1024, 64),
    ((4,), (4,), 128, 8),
    ((2, 1), (3,), 32, 64),
]
FLAGS = list(itertools.product((False, True), repeat=3))
CASES = list(itertools.product(SYSTEMS, FLAGS))


class TriangularSolveBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "A batch, B batch, n, nrhs, upper, transpose, unitriangular"
    IS_OUT = False

    def _time_callable(self, fn, xs):
        # Exclude first-use compilation/library initialization from the adaptive
        # iteration estimate. Otherwise slow compilation can reduce a complete
        # multi-kernel operator measurement to a single timed call.
        fn()
        base.torch_device_fn.synchronize()
        return super()._time_callable(fn, xs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_case_iter(self, dtype):
        for ordinal, (system, flags) in enumerate(self.shapes):
            a_batch, b_batch, n, nrhs = system
            batch = torch.broadcast_shapes(a_batch, b_batch)
            params = dict(zip(("upper", "transpose", "unitriangular"), flags))
            params["includes_full_A_clone"] = True
            shapes = {"B": b_batch + (n, nrhs), "A": a_batch + (n, n)}
            if self.IS_OUT:
                shapes.update(X=batch + (n, nrhs), M=batch + (n, n))
                params["out_layout"] = "column_major"
            yield self._case_from_plan(
                dtype,
                ordinal,
                BenchmarkCasePlan(
                    shape=shapes, params=params, builder_args=(system, flags)
                ),
            )

    def build_inputs(self, case):
        system, flags = case.builder_args[0].builder_args
        a_batch, b_batch, n, nrhs = system
        A = torch.empty(*a_batch, n, n, dtype=case.dtype, device=self.device)
        A.uniform_(-1, 1)
        A.mul_(0.125 / n)
        A.diagonal(0, -2, -1).fill_(7.0 if flags[2] else 2.0)
        B = torch.randn(*b_batch, n, nrhs, dtype=case.dtype, device=self.device)
        if not self.IS_OUT:
            return B, A, *flags
        batch = torch.broadcast_shapes(a_batch, b_batch)
        # Match the functional native output layout, and measure the full
        # solution plus full original-A copy for both native and FlagGems.
        X = torch.empty(
            *batch, nrhs, n, dtype=case.dtype, device=self.device
        ).transpose(-2, -1)
        M = torch.empty(*batch, n, n, dtype=case.dtype, device=self.device).transpose(
            -2, -1
        )
        return B, A, *flags, {"X": X, "M": M}

    def get_input_iter(self, cur_dtype):
        for case in self.get_case_iter(cur_dtype):
            yield self.build_inputs(case)


class TriangularSolveOutBenchmark(TriangularSolveBenchmark):
    IS_OUT = True


@pytest.mark.triangular_solve
@pytest.mark.parametrize("dtype", DTYPES)
def test_triangular_solve(dtype):
    TriangularSolveBenchmark(
        op_name="triangular_solve",
        torch_op=torch.ops.aten.triangular_solve.default,
        gems_op=flag_gems.triangular_solve,
        dtypes=[dtype],
    ).run()


@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("dtype", DTYPES)
def test_triangular_solve_out(dtype):
    TriangularSolveOutBenchmark(
        op_name="triangular_solve_out",
        torch_op=torch.ops.aten.triangular_solve.X,
        gems_op=flag_gems.triangular_solve_out,
        dtypes=[dtype],
    ).run()
