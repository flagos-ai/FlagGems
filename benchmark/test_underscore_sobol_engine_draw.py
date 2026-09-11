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
from torch.quasirandom import SobolEngine

import flag_gems

from . import base

# Sobol draw benchmark shapes: (n_samples, dimension)
SOBOL_SHAPES = [
    (100, 2),
    (1000, 3),
    (10000, 5),
    (100000, 3),
]


class SobolDrawBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOBOL_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, dimension in self.shapes:
            eng = SobolEngine(dimension=dimension, scramble=False)
            quasi = eng.quasi.to(device=self.device)
            sobolstate = eng.sobolstate.to(device=self.device)
            num_generated = 0
            yield (quasi, n, sobolstate, dimension, num_generated, cur_dtype)


@pytest.mark.underscore_sobol_engine_draw
def test_perf_underscore_sobol_engine_draw():
    # Note: torch._sobol_engine_draw has a segfault bug in CUDA (PyTorch 2.12.0+cu130)
    # so we use the gems implementation as the baseline instead
    bench = SobolDrawBenchmark(
        op_name="underscore_sobol_engine_draw",
        torch_op=flag_gems.underscore_sobol_engine_draw,
        dtypes=[torch.float32],
    )
    bench.run()
