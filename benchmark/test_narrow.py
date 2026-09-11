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

# narrow slices along dim 0; enumerate shapes explicitly.
NARROW_SHAPES = [(10000, 256), (10000, 4096), (10000, 65536)]


class NarrowBenchmark(base.Benchmark):
    """Benchmark for narrow operation (zero-copy view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = NARROW_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            dim = 0
            start = shape[dim] // 4
            length = shape[dim] // 2
            yield inp, dim, start, length


@pytest.mark.narrow
def test_narrow():
    bench = NarrowBenchmark(
        op_name="narrow",
        torch_op=torch.narrow,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
