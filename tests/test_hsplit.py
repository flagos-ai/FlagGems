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


@pytest.mark.hsplit
@pytest.mark.parametrize(
    "shape",
    [(128,), (64, 128), (32, 64, 128), (16, 32, 64, 128)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sections", [2, 4])
def test_hsplit_int(shape, dtype, sections, caplog):
    """Test hsplit.int accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.hsplit.int(ref_inp, sections)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.hsplit"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.hsplit.int(inp, sections)
    assert "GEMS HSPLIT" in caplog.text

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)


@pytest.mark.hsplit
@pytest.mark.parametrize(
    "shape",
    [(128,), (64, 128), (32, 64, 128)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("indices", [[32], [16, 48], [20, 40, 80]])
def test_hsplit_array(shape, dtype, indices, caplog):
    """Test hsplit.array accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.hsplit.array(ref_inp, indices)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.hsplit"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.hsplit.array(inp, indices)
    assert "GEMS HSPLIT" in caplog.text

    assert len(res_out) == len(
        ref_out
    ), f"hsplit count mismatch: {len(res_out)} vs {len(ref_out)}"
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_close(utils.to_reference(res), utils.to_reference(ref), dtype)
