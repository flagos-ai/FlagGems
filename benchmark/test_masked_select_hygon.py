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

HYGON_MASKED_SELECT_CASES = [
    ((0,), (0,)),
    ((0, 3), (1, 3)),
    ((1, 3), (0, 3)),
    ((), (0,)),
    ((17,), (17,)),
    ((4097,), (4097,)),
]


class HygonMaskedSelectBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = HYGON_MASKED_SELECT_CASES


def hygon_masked_select_input_fn(case, dtype, device):
    input_shape, mask_shape = case
    inp = torch.zeros(input_shape, dtype=dtype, device=device)
    mask = torch.ones(mask_shape, dtype=torch.bool, device=device)
    yield inp, mask


@pytest.mark.masked_select
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon empty fast path")
def test_masked_select_hygon_empty():
    # Use operator mode: metadata-known empty cases intentionally launch no kernel.
    bench = HygonMaskedSelectBenchmark(
        op_name="masked_select",
        torch_op=torch.masked_select,
        gems_op=flag_gems.masked_select,
        input_fn=hygon_masked_select_input_fn,
        dtypes=[torch.float32],
    )
    bench.run()
