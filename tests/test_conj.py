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

    # Some backends do not implement isclose/equal for complex32 (ComplexHalf,
    # reported as "Unsupported tensor dtype: ComplexHalf"). Compare the real
    # and imaginary lanes as ordinary floats instead, which every backend
    # supports; res_out may be a conj view, and .real/.imag handle that bit.
    utils.gems_assert_close(res_out.real, ref_out.real, res_out.real.dtype)
    utils.gems_assert_close(res_out.imag, ref_out.imag, res_out.imag.dtype)


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

    # Some backends do not implement isclose/equal for complex32 (ComplexHalf,
    # reported as "Unsupported tensor dtype: ComplexHalf"). Compare the real
    # and imaginary lanes as ordinary floats instead, which every backend
    # supports; res_out may be a conj view, and .real/.imag handle that bit.
    utils.gems_assert_close(res_out.real, ref_out.real, res_out.real.dtype)
    utils.gems_assert_close(res_out.imag, ref_out.imag, res_out.imag.dtype)
