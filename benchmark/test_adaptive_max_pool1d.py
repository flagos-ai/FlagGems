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

from benchmark.base import Benchmark


class AdaptiveMaxPool1dBenchmark(Benchmark):
    """Benchmark for adaptive_max_pool1d operator."""

    def set_default_shapes(self):
        """Set default shapes for adaptive_max_pool1d (N, C, L)."""
        self.shapes = [
            (8, 64, 128),
            (16, 128, 256),
            (32, 256, 512),
            (64, 512, 1024),
        ]
        self.shape_desc = "N, C, L"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_shapes()

    def init_user_config(self):
        """Override to prevent loading shapes from YAML."""
        super().init_user_config()
        self.set_default_shapes()

    def get_input_iter(self, cur_dtype):
        """Generate test inputs."""
        for cur_shape in self.shapes:
            N, C, L = cur_shape
            inp = torch.randn(cur_shape, dtype=cur_dtype, device=self.device)
            # Output size is half of input length
            output_size = L // 2
            yield inp, {"output_size": output_size}

    def torch_forward(self, inp, output_size):
        return torch.nn.functional.adaptive_max_pool1d(
            inp, output_size, return_indices=True
        )


@pytest.mark.adaptive_max_pool1d
def test_perf_adaptive_max_pool1d():
    bench = AdaptiveMaxPool1dBenchmark(
        op_name="adaptive_max_pool1d",
        torch_op=torch.ops.aten.adaptive_max_pool1d,
        arg_func=None,
    )
    bench.run()
