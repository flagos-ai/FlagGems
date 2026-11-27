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


@pytest.mark.conj
@pytest.mark.parametrize("shape", [(256,), (32, 64), (2, 3, 4)])
# conj only operates on complex dtypes (FLOAT_DTYPES/INT_DTYPES not applicable)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj(shape, dtype):
    # Create complex input
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj(ref_inp)
    res_out = flag_gems.conj(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.underscore_conj
@pytest.mark.parametrize("shape", [(256,), (32, 64), (2, 3, 4)])
# _conj only operates on complex dtypes (FLOAT_DTYPES/INT_DTYPES not applicable)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test__conj(shape, dtype):
    # Create complex input
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch._conj(ref_inp)
    res_out = flag_gems._conj(inp)

    # Check the original device-side view contract before host observation.
    assert res_out.device == inp.device
    assert res_out.dtype == inp.dtype == dtype
    assert res_out.shape == inp.shape
    assert res_out.stride() == inp.stride()
    assert res_out.storage_offset() == inp.storage_offset()
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out.is_conj() != inp.is_conj()
    # Device complex isclose is unavailable; retain the original tolerance.
    utils.gems_assert_close(res_out.detach().cpu(), ref_out.detach().cpu(), dtype)
