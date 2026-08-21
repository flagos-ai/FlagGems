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

# ``_assert_tensor_metadata`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._assert_tensor_metadata`` and
# ``-m _assert_tensor_metadata`` both work.
setattr(
    pytest.mark,
    "_assert_tensor_metadata",
    MarkDecorator(
        Mark("_assert_tensor_metadata", (), {}, _ispytest=True), _ispytest=True
    ),
)

# Square 2D shapes covering common sizes for the metadata assertion benchmark.
ASSERT_TENSOR_METADATA_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
]


class AssertTensorMetadataBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ASSERT_TENSOR_METADATA_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            size = list(inp.size())
            stride = list(inp.stride())
            yield inp, size, stride, cur_dtype


@pytest.mark.assert_tensor_metadata
def test__assert_tensor_metadata():
    bench = AssertTensorMetadataBenchmark(
        op_name="_assert_tensor_metadata",
        torch_op=torch.ops.aten._assert_tensor_metadata,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
