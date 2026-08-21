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

VSPLIT_CONFIGS = [
    # (shape, indices_or_sections)
    # Integer splits (equal chunks)
    ((4, 6), 2),
    ((8, 4), 4),
    ((12, 8), 3),
    ((16, 16), 8),
    ((20, 10), 5),
    # List splits (custom indices)
    ((4, 6), [1]),
    ((4, 6), [2]),
    ((6, 8), [1, 3]),
    ((8, 4), [2, 4, 6]),
    ((10, 5), [3, 7]),
    # 3D tensors
    ((4, 6, 8), 2),
    ((8, 4, 3), 4),
    ((6, 8, 10), [1, 3]),
    ((10, 5, 7), [3, 7]),
]


@pytest.mark.vsplit
@pytest.mark.parametrize("shape, indices_or_sections", VSPLIT_CONFIGS)
def test_accuracy_vsplit(shape, indices_or_sections, caplog):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(indices_or_sections, int):
        ref_out = torch.ops.aten.vsplit.int(ref_inp, indices_or_sections)
    else:
        ref_out = torch.ops.aten.vsplit.array(ref_inp, indices_or_sections)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.vsplit"):
        with flag_gems.use_gems():
            if isinstance(indices_or_sections, int):
                res_out = torch.ops.aten.vsplit.int(inp, indices_or_sections)
            else:
                res_out = torch.ops.aten.vsplit.array(inp, indices_or_sections)
    assert "GEMS VSPLIT" in caplog.text

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res_chunk, ref_chunk) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(res_chunk, ref_chunk, torch.float32)
