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


class WrappedLinearPrepackBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4, 8), (64, 128), (256, 256), (512, 1024), (2048, 1024)]

    def get_input_iter(self, dtype):
        for N, K in self.shapes:
            weight = torch.randn((N, K), dtype=dtype, device=self.device)
            scale = torch.tensor(0.03125, dtype=dtype, device=self.device)
            zero_point = torch.tensor(-3, dtype=torch.int64, device=self.device)
            bias = torch.randn((N,), dtype=dtype, device=self.device)
            yield weight, scale, zero_point, bias


@pytest.mark.wrapped_linear_prepack
def test_wrapped_linear_prepack_benchmark():
    bench = WrappedLinearPrepackBenchmark(
        op_name="wrapped_linear_prepack",
        torch_op=torch.ops.aten._wrapped_linear_prepack,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems._wrapped_linear_prepack)
    bench.run()
