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

# Square 2D shapes covering common sizes for view benchmark
UNSAFE_VIEW_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


class UnsafeViewBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = UNSAFE_VIEW_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            new_shape = (shape[0] * shape[1],)
            yield inp, new_shape


@pytest.mark.unsafe_view
def test_unsafe_view():
    bench = UnsafeViewBenchmark(
        op_name="unsafe_view",
        torch_op=torch.ops.aten._unsafe_view,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
