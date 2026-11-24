# Copyright 2026, The FlagOS Contributors.
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

# cdist on CUDA does not support float16/bfloat16, so only float32 is exercised.
EUC_BACKWARD_SHAPES = [
    ((64, 128), (64, 128)),
    ((128, 256), (128, 256)),
]


class EuclideanDistBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EUC_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for (n1, d), (n2, d2) in self.shapes:
            x1 = torch.randn(n1, d, dtype=cur_dtype, device=self.device)
            x2 = torch.randn(n2, d2, dtype=cur_dtype, device=self.device)
            grad = torch.randn(n1, n2, dtype=cur_dtype, device=self.device)
            yield grad, x1, x2


@pytest.mark.euclidean_dist_backward
def test_euclidean_dist_backward():
    bench = EuclideanDistBackwardBenchmark(
        op_name="euclidean_dist_backward",
        torch_op=flag_gems.euclidean_dist_backward,
        dtypes=[torch.float32],
    )
    bench.run()
