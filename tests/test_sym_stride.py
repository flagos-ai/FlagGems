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

# Test shapes for sym_stride - covering various tensor dimensionalities
SYM_STRIDE_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


@pytest.mark.sym_stride
@pytest.mark.parametrize("shape", SYM_STRIDE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sym_stride(shape, dtype):
    """Test sym_stride operator accuracy."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sym_stride(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.sym_stride(inp)

    # Compare stride results (convert to tensors for gems_assert_equal)
    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))
