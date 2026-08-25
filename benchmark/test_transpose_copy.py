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

TRANSPOSE_COPY_CASES = [
    ((64, 128), 0, 1, False),
    ((256, 512), 0, -1, False),
    ((64, 128, 256), 0, 2, False),
    ((8, 16, 32, 64), 1, -1, False),
    ((256, 512), 0, 0, False),
    ((128, 256, 64), 0, -1, True),
]


class TransposeCopyBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = TRANSPOSE_COPY_CASES

    def get_input_iter(self, dtype):
        for shape, dim0, dim1, non_contiguous in self.shapes:
            if non_contiguous:
                base_shape = (shape[0] * 2, *shape[1:])
                input = torch.empty(base_shape, dtype=dtype, device=self.device)[::2]
            else:
                input = torch.empty(shape, dtype=dtype, device=self.device)
            yield input, dim0, dim1


@pytest.mark.transpose_copy
def test_transpose_copy():
    bench = TransposeCopyBenchmark(
        op_name="transpose_copy",
        torch_op=torch.ops.aten.transpose_copy.int,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
