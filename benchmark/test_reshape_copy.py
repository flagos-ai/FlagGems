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

# (input_shape, size, transpose) triples; each `size` describes the shape of the
# contiguous copy that _reshape_copy materializes. `transpose=True` makes the
# input non-contiguous so the strided gather path is exercised.
RESHAPE_COPY_SHAPES = [
    ((1024, 1024), [1048576], False),
    ((2048, 2048), [2048, 2048], False),
    ((4096, 4096), [16777216], False),
    ((8192, 8192), [8192, 8192], False),
    ((1024, 1024), [2, 524288], False),
    ((1024, 1024), [1048576], True),
]


class ReshapeCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RESHAPE_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for input_shape, size, transpose in self.shapes:
            inp = torch.randn(input_shape, dtype=cur_dtype, device=self.device)
            if transpose:
                inp = inp.t()
            yield inp, size


@pytest.mark.reshape_copy
def test_reshape_copy():
    bench = ReshapeCopyBenchmark(
        op_name="_reshape_copy",
        torch_op=torch.ops.aten._reshape_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
