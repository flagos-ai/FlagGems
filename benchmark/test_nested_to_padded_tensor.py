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

import numpy as np
import pytest
import torch

from . import base, consts

# Nested to padded tensor benchmark shapes: (batch_size, max_rows, trailing_dim)
# trailing_dim kept moderate (32/64) so one row fits in a single kernel block.
NESTED_TO_PADDED_SHAPES = [
    (8, 8, 32),
    (16, 16, 64),
    (32, 32, 64),
    (64, 64, 64),
    (128, 64, 64),
    (256, 128, 64),
    (512, 256, 32),
]


class NestedToPaddedTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_TO_PADDED_SHAPES

    def record_shapes(self, *args, **kwargs):
        # The first arg is a nested tensor; NestedTensorImpl does not support
        # .size(), so fall back to recording its component sizes instead of the
        # base implementation which would call .size() and raise.
        nt = args[0]
        return [nt._nested_tensor_size().tolist()] + list(args[1:])

    def get_input_iter(self, cur_dtype):
        for batch_size, max_rows, trailing_dim in self.shapes:
            # Force component 0 to the full row count so the inferred padded
            # shape is deterministic; the rest are ragged.
            np.random.seed(42)
            row_counts = np.random.randint(1, max_rows + 1, size=batch_size)
            row_counts[0] = max_rows
            comps = [
                torch.randn([int(r), trailing_dim], dtype=cur_dtype, device=self.device)
                for r in row_counts
            ]
            nt = torch.nested.nested_tensor(comps, device=self.device)

            yield nt, 0.0, None


@pytest.mark.nested_to_padded_tensor
def test_nested_to_padded_tensor():
    bench = NestedToPaddedTensorBenchmark(
        op_name="nested_to_padded_tensor",
        torch_op=torch.ops.aten.nested_to_padded_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
