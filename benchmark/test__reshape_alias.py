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

from . import base, consts

# ``_reshape_alias`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._reshape_alias`` and ``-m
# _reshape_alias`` both work.
setattr(
    pytest.mark,
    "_reshape_alias",
    MarkDecorator(Mark("_reshape_alias", (), {}, _ispytest=True), _ispytest=True),
)

# Square 2D shapes covering common sizes for view benchmark
RESHAPE_ALIAS_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


class ReshapeAliasBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RESHAPE_ALIAS_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            size = [shape[0] * shape[1]]
            stride = [1]
            yield inp, size, stride


@pytest.mark._reshape_alias
def test__reshape_alias():
    bench = ReshapeAliasBenchmark(
        op_name="_reshape_alias",
        torch_op=torch.ops.aten._reshape_alias,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
