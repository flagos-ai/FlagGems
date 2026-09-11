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

from . import base, consts

# Benchmark shapes for sym_stride - covering various tensor dimensionalities
SYM_STRIDE_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


class SymStrideBenchmark(base.Benchmark):
    """Custom benchmark for sym_stride - returns tensor metadata (stride), not a computed tensor."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = SYM_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (x,)


@pytest.mark.sym_stride
def test_sym_stride():
    bench = SymStrideBenchmark(
        op_name="sym_stride",
        torch_op=torch.ops.aten.sym_stride,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
