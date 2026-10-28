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

# One (dimension, MAXBIT) state buffer per shape; the operator fills the whole
# buffer from a fixed table, so the runtime scales with the dimension count.
SOBOL_INIT_SHAPES = [
    (100, 30),
    (500, 30),
    (1000, 30),
    (5000, 30),
]

MAXBIT = 30


def sobol_init_input_fn(shape, dtype, device):
    dimension = shape[0]
    state = torch.zeros((dimension, MAXBIT), dtype=torch.int64, device=device)
    yield state, dimension


class SobolInitBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOBOL_INIT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            dimension = shape[0]
            state = torch.zeros(
                (dimension, MAXBIT), dtype=torch.int64, device=self.device
            )
            yield state, dimension


@pytest.mark.sobol_engine_initialize_state_
def test_sobol_engine_initialize_state_perf():
    # Note: aten's _sobol_engine_initialize_state_ has no CUDA kernel (calling it
    # on a device tensor aborts the process), so the FlagGems implementation is
    # used as the baseline; the benchmark documents its latency/speedup record.
    bench = SobolInitBenchmark(
        op_name="sobol_engine_initialize_state_",
        torch_op=flag_gems._sobol_engine_initialize_state_,
        dtypes=[torch.int64],
    )
    bench.set_gems(flag_gems._sobol_engine_initialize_state_)
    bench.run()
