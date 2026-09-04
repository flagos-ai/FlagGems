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

from .performance_utils import GenericBenchmark


class Benchmark(GenericBenchmark):
    """Benchmark for _sobol_engine_initialize_state_ operator."""

    # Note: This operator is not compute-intensive (it's a lookup table fill operation)
    # Benchmarking serves mainly to verify no regression vs native implementation
    DEFAULT_SHAPES = [(100, 30), (500, 30), (1000, 30), (5000, 30)]
    DEFAULT_METRICS = ["latency", "speedup"]

    def set_more_shapes(self):
        """Define benchmark shapes: (dimension, 30) where dimension varies."""
        self.shapes = [
            {"dimension": 10},
            {"dimension": 50},
            {"dimension": 100},
            {"dimension": 500},
            {"dimension": 1000},
            {"dimension": 5000},
            {"dimension": 10000},
        ]

    def get_input_iter(self, cur_shape):
        """Generate input tensors for benchmarking."""
        dimension = cur_shape["dimension"]
        # State tensor must be zeros initially
        state = torch.zeros((dimension, 30), dtype=torch.int64, device=self.device)
        yield state.clone(), dimension

    @pytest.mark.sobol_engine_initialize_state_
    def test_perf_sobol_engine_initialize_state_(self):
        """Run performance benchmark."""
        self.op_name = "sobol_engine_initialize_state_"
        self.torch_op = torch.ops.aten._sobol_engine_initialize_state_
        self.run_benchmark()
