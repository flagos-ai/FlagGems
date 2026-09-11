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


class BroadcastToBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Fixed (src_shape, target_shape) pairs chosen to exercise distinct broadcast
        # patterns: leading-dim insertion, mid/leading axis expansion, and full-rank
        # expansion, so the kernel covers each broadcast code path.
        self.shapes = [
            ((1024,), (1, 1024)),  # 1D -> 2D broadcast (add leading dim)
            ((64, 1), (64, 4096)),  # 2D -> 2D broadcast (expand dim 1)
            ((1, 64), (4096, 64)),  # 2D -> 2D broadcast (expand dim 0)
            ((1, 1, 1), (64, 512, 512)),  # 3D -> 3D broadcast (expand all dims)
        ]
        self.shape_desc = "src_shape -> target_shape"

    def get_input_iter(self, dtype):
        for src_shape, target_shape in self.shapes:
            x = base.generate_tensor_input(src_shape, dtype, self.device)
            yield (x, target_shape)


@pytest.mark.broadcast_to
def test_broadcast_to():
    benchmark = BroadcastToBenchmark(
        op_name="broadcast_to",
        torch_op=torch.broadcast_to,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()
