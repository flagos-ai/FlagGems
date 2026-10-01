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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base

# ``_sobol_engine_scramble_`` starts with an underscore and ``pytest.mark``
# refuses attribute access for such names; register the marker directly on the
# MarkGenerator (same pattern as the test module).
setattr(
    pytest.mark,
    "_sobol_engine_scramble_",
    MarkDecorator(
        Mark("_sobol_engine_scramble_", (), {}, _ispytest=True), _ispytest=True
    ),
)

# (dimension, MAXBIT) shapes; the operator scrambles one (MAXBIT, MAXBIT) block
# per dimension, so the runtime scales with the dimension count.
SOBOL_SCRAMBLE_SHAPES = [
    (100, 30),
    (500, 30),
    (1000, 30),
    (5000, 30),
]

MAXBIT = 30


def _make_args(dimension, device):
    sobolstate = torch.randint(
        0, 2, (dimension, MAXBIT), dtype=torch.long, device=device
    )
    ltm = torch.randint(
        0, 2, (dimension, MAXBIT, MAXBIT), dtype=torch.long, device=device
    ).tril()
    return sobolstate, ltm, dimension


def sobol_scramble_input_fn(shape, dtype, device):
    # aten has no CUDA kernel and the op mutates its input in place, so each
    # invocation gets a fresh pair of tensors; dtype is fixed (int64 state).
    dimension = shape[0]
    yield _make_args(dimension, device)


class SobolScrambleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SOBOL_SCRAMBLE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield _make_args(shape[0], self.device)


@pytest.mark._sobol_engine_scramble_
def test_sobol_engine_scramble_perf():
    # Note: aten's _sobol_engine_scramble_ has no CUDA kernel and the op mutates
    # its input in place, so the FlagGems implementation is used as the baseline
    # (same approach as the sibling sobol draw benchmark).
    bench = SobolScrambleBenchmark(
        op_name="sobol_engine_scramble_",
        torch_op=flag_gems._sobol_engine_scramble_,
        dtypes=[torch.int64],
    )
    bench.set_gems(flag_gems._sobol_engine_scramble_)
    bench.run()
