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


def _reference_linalg_polar(inp):
    svd_U, singular_values, Vh = torch.linalg.svd(inp, full_matrices=False)
    polar_U = svd_U @ Vh
    polar_H = Vh.mH @ (singular_values.unsqueeze(-1) * Vh)
    return polar_U, 0.5 * (polar_H + polar_H.mH)


def _reference_linalg_polar_out(inp, *, U, H):
    result_U, result_H = _reference_linalg_polar(inp)
    U.resize_as_(result_U).copy_(result_U)
    H.resize_as_(result_H).copy_(result_H)
    return U, H


if hasattr(torch.ops.aten, "linalg_polar"):
    TORCH_LINALG_POLAR = torch.ops.aten.linalg_polar.default
    TORCH_LINALG_POLAR_OUT = torch.ops.aten.linalg_polar.out
else:
    TORCH_LINALG_POLAR = _reference_linalg_polar
    TORCH_LINALG_POLAR_OUT = _reference_linalg_polar_out


class LinalgPolarBenchmark(base.Benchmark):
    CORE_SHAPES = [(4, 2), (8, 4), (16, 8), (16, 8, 4)]
    COMPREHENSIVE_SHAPES = [(64, 8, 4), (32, 16, 8)]

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.COMPREHENSIVE_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=dtype, device=self.device),)


class LinalgPolarOutBenchmark(LinalgPolarBenchmark):
    def get_input_iter(self, dtype):
        for (inp,) in super().get_input_iter(dtype):
            U = torch.empty_like(inp)
            H = torch.empty(
                (*inp.shape[:-2], inp.shape[-1], inp.shape[-1]),
                dtype=dtype,
                device=self.device,
            )
            yield inp, {"U": U, "H": H}


@pytest.mark.linalg_polar
def test_linalg_polar():
    bench = LinalgPolarBenchmark(
        op_name="linalg_polar",
        torch_op=TORCH_LINALG_POLAR,
        gems_op=flag_gems.linalg_polar,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.linalg_polar_out
def test_linalg_polar_out():
    bench = LinalgPolarOutBenchmark(
        op_name="linalg_polar_out",
        torch_op=TORCH_LINALG_POLAR_OUT,
        gems_op=flag_gems.linalg_polar_out,
        dtypes=[torch.float32],
    )
    bench.run()
