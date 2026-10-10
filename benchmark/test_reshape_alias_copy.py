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

# (input_shape, size, stride) triples; `size`/`stride` describe the view that is
# copied into fresh contiguous storage.
RESHAPE_ALIAS_COPY_SHAPES = [
    ((1024, 1024), [1048576], [1]),
    ((2048, 2048), [2048, 2048], [2048, 1]),
    ((4096, 4096), [16777216], [1]),
    ((8192, 8192), [8192, 8192], [8192, 1]),
    ((1024, 1024), [1024, 1024], [1, 1024]),
]


class ReshapeAliasCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RESHAPE_ALIAS_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for input_shape, size, stride in self.shapes:
            inp = torch.randn(input_shape, dtype=cur_dtype, device=self.device)
            yield inp, size, stride


@pytest.mark.reshape_alias_copy
def test_reshape_alias_copy():
    bench = ReshapeAliasCopyBenchmark(
        op_name="_reshape_alias_copy",
        torch_op=torch.ops.aten._reshape_alias_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
