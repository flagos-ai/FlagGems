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


class NestedTensorFromMaskBenchmark(base.Benchmark):
    """
    Benchmark for _nested_tensor_from_mask, which builds a nested tensor from a
    padded dense tensor (N, L, D) and a boolean padding mask (N, L).
    """

    def set_shapes(self, shape_file_path=None):
        # (N, L, D) padded shapes spanning small-to-large batch/length/feature.
        self.shapes = [
            (16, 64, 64),
            (32, 128, 64),
            (64, 256, 128),
            (128, 512, 128),
        ]

    def get_input_iter(self, cur_dtype):
        for N, L, D in self.shapes:
            t = torch.randn(N, L, D, dtype=cur_dtype, device=self.device)
            lengths = torch.randint(0, L + 1, (N,), device=self.device)
            idx = torch.arange(L, device=self.device).unsqueeze(0)
            mask = idx < lengths.unsqueeze(1)
            yield t, mask


@pytest.mark.nested_tensor_from_mask
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_tensor_from_mask(dtype):
    bench = NestedTensorFromMaskBenchmark(
        op_name="nested_tensor_from_mask",
        torch_op=torch._nested_tensor_from_mask,
        dtypes=[dtype],
    )
    bench.run()
