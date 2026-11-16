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

# (input_shape, dim, split_size, transpose) tuples. dim=0 splits contiguous
# row blocks, dim=1 exercises the strided gather path, `transpose=True` makes
# the input non-contiguous.
SPLIT_COPY_SHAPES = [
    ((2048, 8192), 0, 512, False),
    ((8192, 8192), 0, 2048, False),
    ((8, 8192, 8192), 0, 2048, False),
    ((8192, 8192), 1, 2048, False),
    ((16, 8192, 4096), 1, 1024, False),
    ((8192, 8192), 0, 2048, True),
]


class SplitCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPLIT_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for input_shape, dim, split_size, transpose in self.shapes:
            inp = torch.randn(input_shape, dtype=cur_dtype, device=self.device)
            if transpose:
                inp = inp.t()
            yield inp, split_size, dim


@pytest.mark.split_copy
def test_split_copy():
    bench = SplitCopyBenchmark(
        op_name="split_copy",
        torch_op=torch.ops.aten.split_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
