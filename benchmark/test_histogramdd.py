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

# (num_samples, D) cases; the bin count per dimension matches D.
HISTOGRAMDD_SHAPES = [(4096, 2), (4096, 3), (4096, 4)]


# Native histogramdd is CPU-only; the baseline runs on a CPU copy of the input.
def _histogramdd_baseline(inp, bins, range=None, weight=None, density=False):
    # CPU histogramdd lacks a Half kernel; compute the baseline in float32.
    return torch.histogramdd(
        inp.float().cpu(),
        bins=bins,
        range=range,
        weight=weight,
        density=density,
    )


def _input_fn(shape, dtype, device):
    bins = [4] * shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"bins": bins}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, {"bins": bins, "density": True}


class HistogramddBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = HISTOGRAMDD_SHAPES
        self.shape_desc = "num_samples, D"


@pytest.mark.histogramdd
def test_benchmark_histogramdd():
    bench = HistogramddBenchmark(
        op_name="histogramdd",
        torch_op=_histogramdd_baseline,
        gems_op=flag_gems.histogramdd,
        input_fn=_input_fn,
        # The kernel and the CPU baseline both support float32/float64 only.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
