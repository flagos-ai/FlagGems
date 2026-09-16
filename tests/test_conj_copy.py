# Copyright 2026, The FlagOS Contributors.
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


@pytest.mark.conj_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# _conj_copy only operates on complex dtypes (FLOAT_DTYPES/INT_DTYPES not applicable)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy(shape, dtype):
    # Create complex input from independent real/imaginary parts.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch._conj_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch._conj_copy(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conj_copy_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy_out(shape, dtype):
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.ops.aten._conj_copy.out(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.ops.aten._conj_copy.out(inp, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.conj_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_copy_with_conj_bit(shape, dtype):
    # _conj_copy of an already-conjugated tensor should yield the original
    # values: conj(conj(x)) == x.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    base = torch.complex(real, imag).to(dtype)
    inp = base.conj()  # logical conj view, conj bit set
    ref_inp = utils.to_reference(inp)

    ref_out = torch._conj_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch._conj_copy(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
