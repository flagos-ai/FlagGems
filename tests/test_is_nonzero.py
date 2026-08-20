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


@pytest.mark.is_nonzero
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
def test_is_nonzero(dtype):
    # Test non-zero values
    inp = torch.tensor([1], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.is_nonzero(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.is_nonzero(inp)
    assert res_out == ref_out, f"Expected {ref_out}, got {res_out}"

    # Test zero values
    inp_zero = torch.tensor([0], dtype=dtype, device=flag_gems.device)
    ref_inp_zero = utils.to_reference(inp_zero)
    ref_out_zero = torch.is_nonzero(ref_inp_zero)
    with flag_gems.use_gems():
        res_out_zero = torch.is_nonzero(inp_zero)
    assert res_out_zero == ref_out_zero, f"Expected {ref_out_zero}, got {res_out_zero}"


@pytest.mark.is_nonzero
def test_is_nonzero_exception():
    # Test that multi-element tensor raises RuntimeError
    inp = torch.tensor([1, 2, 3], device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.is_nonzero(inp)
