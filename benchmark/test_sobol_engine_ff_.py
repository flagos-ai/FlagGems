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

# The native aten op only supports CPU (it segfaults on CUDA), so the baseline
# is a composed aten equivalent and the native baseline is skipped.
DIMENSIONS = [16, 64, 256]
N_VALUES = [512, 4096]
MAXBIT = 30


def _sobol_ff_baseline(quasi, n, sobolstate, dimension, num_generated):
    # The native aten op only supports CPU (it segfaults on CUDA).
    quasi = quasi.cpu().clone()
    sobolstate = sobolstate.cpu().clone()
    torch._sobol_engine_ff_(quasi, n, sobolstate, dimension, num_generated)
    return quasi


def _input_fn(case, dtype, device):
    dimension, n, num_generated = case
    quasi = torch.zeros(dimension, dtype=torch.long, device=device)
    sobolstate = torch.randint(
        0, 2**30, (dimension, MAXBIT), dtype=torch.long, device=device
    )
    yield quasi, n, sobolstate, dimension, num_generated


class SobolEngineFfBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [(d, n, g) for d in DIMENSIONS for n in N_VALUES for g in [n]]
    DEFAULT_SHAPE_DESC = "dimension, n, num_generated"

    def init_default_config(self):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        # Each case bundles dimension, n and num_generated; shape-only
        # configuration files are not compatible with this benchmark.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for case in self.shapes:
            yield from _input_fn(case, dtype, self.device)

    def supports_cases(self):
        # Custom per-case iteration; the generic case machinery does not apply.
        return False


@pytest.mark.sobol_engine_ff_
def test_benchmark_sobol_engine_ff_():
    bench = SobolEngineFfBenchmark(
        op_name="sobol_engine_ff_",
        torch_op=_sobol_ff_baseline,
        gems_op=flag_gems._sobol_engine_ff_,
        input_fn=_input_fn,
        dtypes=[torch.int64],
    )
    bench.run()
