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

from . import base

# `_fill_mem_eff_dropout_mask_` only accepts 4D float32 tensors of shape
# (batch, heads, queries, keys); representative attention mask shapes.
# The ATen op is hard-coded to float32 (curand uniform), so only float32 is benchmarked.
MASK_DTYPES = [torch.float32]


def _fill_mem_eff_dropout_mask_input_fn(shape, dtype, device):
    mask = torch.empty(shape, dtype=dtype, device=device)
    # dropout_p is unused by the kernel but kept for API compatibility.
    yield mask, {"dropout_p": 0.0, "seed": 42, "offset": 0}


class FillMemEffDropoutMaskBenchmark(base.GenericBenchmark4DOnly):
    DEFAULT_SHAPE_DESC = "B, H, Q, K"

    def set_more_shapes(self):
        return []

    def init_user_config(self):
        super().init_user_config()
        # The kernel indexes flat elements with int32 arithmetic, so cap total
        # numel below 2**31 to avoid overflow on the largest 4D shapes.
        self.shapes = [s for s in self.shapes if 1 < s[0] * s[1] * s[2] * s[3] < 2**31]


@pytest.mark.fill_mem_eff_dropout_mask_
def test_fill_mem_eff_dropout_mask_():
    bench = FillMemEffDropoutMaskBenchmark(
        op_name="fill_mem_eff_dropout_mask_",
        input_fn=_fill_mem_eff_dropout_mask_input_fn,
        torch_op=torch.ops.aten._fill_mem_eff_dropout_mask_,
        dtypes=MASK_DTYPES,
    )
    bench.run()
