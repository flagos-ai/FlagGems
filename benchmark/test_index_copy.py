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

INDEX_COPY_BACKWARD_CASES = [
    ((1024, 64), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
]


class IndexCopyBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [case[0] for case in INDEX_COPY_BACKWARD_CASES]


def _tensor_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.index_copy_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_copy_backward():
    bench = IndexCopyBackwardBenchmark(
        input_fn=_tensor_input_fn,
        op_name="index_copy",
        torch_op=torch.index_copy,
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.run()
