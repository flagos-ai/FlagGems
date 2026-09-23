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

# Test shapes for sym_storage_offset - covering various tensor dimensionalities
SYM_STORAGE_OFFSET_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


@pytest.mark.sym_storage_offset
@pytest.mark.parametrize("shape", SYM_STORAGE_OFFSET_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sym_storage_offset(shape, dtype, caplog):
    """Test sym_storage_offset operator accuracy."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sym_storage_offset(ref_inp)
    with flag_gems.use_gems():
        with caplog.at_level("DEBUG", logger="flag_gems.ops.sym_storage_offset"):
            res_out = torch.ops.aten.sym_storage_offset(inp)

    assert "GEMS SYM_STORAGE_OFFSET" in caplog.text
    # Compare storage offset results (convert to tensors for gems_assert_equal)
    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))
