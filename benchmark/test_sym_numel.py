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

# Benchmark shapes for sym_numel - covering various tensor dimensionalities
SYM_NUMEL_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


class SymNumelBenchmark(base.Benchmark):
    """Custom benchmark for sym_numel - returns tensor metadata (numel), not a computed tensor."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SYM_NUMEL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=flag_gems.device)
            yield inp,

    def get_bwd_input_iter(self, cur_dtype):
        # sym_numel has no backward pass (returns scalar/int, not differentiable)
        pass

    def set_more_shapes(self):
        # No additional shapes needed for metadata operation
        pass


@pytest.mark.sym_numel
@pytest.mark.parametrize(
    "op_name, torch_op, dtype",
    [("sym_numel", torch.ops.aten.sym_numel, torch.float32)],
)
def test_perf_sym_numel(op_name, torch_op, dtype):
    bench = SymNumelBenchmark(
        op_name=op_name, torch_op=torch_op, dtypes=[dtype], mode="fwd"
    )
    bench.run()
