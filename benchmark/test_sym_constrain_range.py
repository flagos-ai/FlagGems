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

# Scalar sizes for sym_constrain_range - it operates on a scalar symbolic size,
# not a tensor, so we sweep representative magnitudes rather than tensor shapes.
SYM_CONSTRAIN_RANGE_SIZES = [1, 16, 256, 4096, 65536]


class SymConstrainRangeBenchmark(base.Benchmark):
    """Custom benchmark for sym_constrain_range - validates a scalar bound with a
    Triton kernel and returns nothing (void), so inputs are scalar sizes rather
    than tensors."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SYM_CONSTRAIN_RANGE_SIZES

    def get_input_iter(self, cur_dtype):
        for size in self.shapes:
            # min/max chosen so the value is always in range (no exception path)
            yield (size, {"min": 0, "max": size})


@pytest.mark.sym_constrain_range
def test_sym_constrain_range():
    bench = SymConstrainRangeBenchmark(
        op_name="sym_constrain_range",
        torch_op=torch.ops.aten.sym_constrain_range,
        # torch.ops.aten dispatch may bypass FlagGems for this op, so point the
        # gems side directly at the FlagGems Triton implementation.
        gems_op=flag_gems.sym_constrain_range,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
