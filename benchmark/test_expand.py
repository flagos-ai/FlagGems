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

# Shapes with at least one dimension of size 1 for valid expand targets.
# Expand can only broadcast from dimensions of size 1 to larger values.
EXPAND_SHAPES = [
    (2, 1),
    (1, 3),
    (2, 1, 3),
    (1, 1, 1),
    (1,),
    (1, 2),
    (128, 1),
    (1, 512),
    (64, 1, 64),
    (1, 256, 256),
]


class ExpandBenchmark(base.Benchmark):
    """Benchmark for expand operation (zero-copy view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = EXPAND_SHAPES

    def get_input_iter(self, dtype):
        # Expansion factors for deterministic benchmark
        factors = [2, 3, 4]
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            input_shape = list(inp.shape)
            target_shape = list(input_shape)
            # Expand dimensions that are 1 using a fixed cycle of factors
            for i in range(len(target_shape)):
                if input_shape[i] == 1:
                    factor_idx = len(
                        [j for j in range(i) if input_shape[j] == 1]
                    ) % len(factors)
                    target_shape[i] = input_shape[i] * factors[factor_idx]
            yield inp, target_shape


@pytest.mark.expand
def test_expand():
    bench = ExpandBenchmark(
        op_name="expand",
        torch_op=torch.ops.aten.expand,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.expand_
def test_expand_():
    bench = ExpandBenchmark(
        op_name="expand_",
        torch_op=torch.Tensor.expand,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
