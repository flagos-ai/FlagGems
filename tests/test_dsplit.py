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

from . import accuracy_utils as utils

DSPLIT_CONFIGS = [
    # (shape, indices_or_sections)
    # Integer splits (equal chunks)
    ((4, 6, 8), 2),
    ((8, 4, 12), 4),
    ((12, 8, 16), 4),
    ((16, 16, 8), 2),
    ((20, 10, 15), 5),
    # List splits (custom indices)
    ((4, 6, 8), [2]),
    ((4, 6, 8), [4]),
    ((6, 8, 10), [2, 6]),
    ((8, 4, 12), [3, 6, 9]),
    ((10, 5, 15), [5, 10]),
    # 4D tensors
    ((4, 6, 8, 3), 2),
    ((8, 4, 12, 5), 4),
    ((6, 8, 10, 7), [2, 6]),
    ((10, 5, 15, 3), [5, 10]),
]


@pytest.mark.dsplit
@pytest.mark.parametrize("shape, indices_or_sections", DSPLIT_CONFIGS)
def test_accuracy_dsplit(shape, indices_or_sections):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(indices_or_sections, int):
        ref_out = torch.ops.aten.dsplit.int(ref_inp, indices_or_sections)
    else:
        ref_out = torch.ops.aten.dsplit.array(ref_inp, indices_or_sections)

    res_out = flag_gems.dsplit(inp, indices_or_sections)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res_chunk, ref_chunk) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(res_chunk, ref_chunk, torch.float32)
