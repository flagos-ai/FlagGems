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

from . import base, consts

# Source shapes chosen to cover a 1-D case, a mid-rank case, and a higher-rank
# case; each is broadcast to a target that prepends a new sparse dimension.
SPARSE_BROADCAST_TO_SHAPES = [
    (256,),
    (20, 320, 15),
    (16, 128, 64, 60),
]


def _make_sparse_coo(shape, dtype, device, density=0.3):
    """Build a coalesced sparse COO tensor on the target device."""
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < density
    return (x * mask).to_sparse().coalesce()


class SparseBroadcastToBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_BROADCAST_TO_SHAPES
        self.shape_desc = "src_shape -> target_shape"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = _make_sparse_coo(shape, cur_dtype, self.device)
            yield x, (4,) + shape  # prepend a new sparse dimension of size 4


@pytest.mark.sparse_broadcast_to
def test_sparse_broadcast_to():
    bench = SparseBroadcastToBenchmark(
        op_name="sparse_broadcast_to",
        torch_op=torch.ops.aten._sparse_broadcast_to,
        gems_op=flag_gems.sparse_broadcast_to,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
