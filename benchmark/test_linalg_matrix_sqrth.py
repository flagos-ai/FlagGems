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

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops.aten, "linalg_matrix_sqrth"),
    reason=(
        "linalg_matrix_sqrth requires PyTorch nightly 2026-06-27 or newer, "
        "or PyTorch 2.15+"
    ),
)

CORE_SHAPES = [
    (1, 1),
    (8, 8),
    (32, 32),
    (64, 64),
    (128, 128),
    (4, 32, 32),
]
COMPREHENSIVE_SHAPES = [
    (256, 256),
    (2, 128, 128),
    (4, 64, 64),
]
DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    DTYPES += [torch.float64, torch.complex128]


def _make_hpd(shape, dtype, device):
    matrix = torch.randn(shape, dtype=dtype, device=device)
    n = shape[-1]
    return matrix @ matrix.mH + 0.5 * torch.eye(n, dtype=dtype, device=device)


class MatrixSqrthBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.set_more_shapes()

    def set_more_shapes(self):
        return list(COMPREHENSIVE_SHAPES)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield (_make_hpd(shape, cur_dtype, self.device),)


class MatrixSqrthOutBenchmark(MatrixSqrthBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = _make_hpd(shape, cur_dtype, self.device)
            out = torch.empty_like(inp)
            yield inp, {"out": out}


@pytest.mark.linalg_matrix_sqrth
def test_linalg_matrix_sqrth():
    bench = MatrixSqrthBenchmark(
        op_name="linalg_matrix_sqrth",
        torch_op=torch.ops.aten.linalg_matrix_sqrth.default,
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out():
    bench = MatrixSqrthOutBenchmark(
        op_name="linalg_matrix_sqrth_out",
        torch_op=torch.ops.aten.linalg_matrix_sqrth.out,
        dtypes=DTYPES,
    )
    bench.run()
