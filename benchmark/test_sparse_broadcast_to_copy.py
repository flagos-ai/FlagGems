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

# ``aten::_sparse_broadcast_to_copy`` has no native sparse kernel, so the
# broadcast semantics of ``aten::_sparse_broadcast_to`` are the baseline the
# copy implementation must match; the two produce identical outputs.
SPARSE_BROADCAST_TO_COPY_SHAPES = [
    ((256,), (4, 256)),
    ((20, 320, 15), (4, 20, 320, 15)),
    ((1024, 1024), (4, 1024, 1024)),
]


class SparseBroadcastToCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_BROADCAST_TO_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for src_shape, target_shape in self.shapes:
            x = torch.randn(src_shape, dtype=cur_dtype, device=self.device)
            mask = torch.rand(src_shape, device=self.device) < 0.3
            inp = (x * mask).to_sparse().coalesce()
            yield inp, target_shape


@pytest.mark.sparse_broadcast_to_copy
def test_sparse_broadcast_to_copy():
    """Benchmark the copy variant against ``aten::_sparse_broadcast_to``."""
    bench = SparseBroadcastToCopyBenchmark(
        op_name="sparse_broadcast_to_copy",
        torch_op=torch.ops.aten._sparse_broadcast_to,
        gems_op=flag_gems.sparse_broadcast_to_copy,
        dtypes=[torch.float32],
    )
    bench.run()
