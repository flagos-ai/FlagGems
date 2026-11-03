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


def _assert_conj_close(res, ref, dtype):
    # torch-xpu (kunlunxin) eager has no complex32 arithmetic and compares a
    # physical result against a lazy conj-bit reference, so compare on CPU with
    # the conj bit resolved (complex32 upcast to complex64). This only works
    # around the vendor reference/comparison gap; kernel values are correct.
    if flag_gems.vendor_name == "kunlunxin" and dtype in (
        torch.complex32,
        torch.complex64,
    ):
        cmp = torch.complex64 if dtype == torch.complex32 else dtype
        r = res.detach().cpu().to(cmp).resolve_conj()
        f = ref.detach().cpu().to(cmp).resolve_conj()
        torch.testing.assert_close(r, f, rtol=1e-3, atol=1e-3)
    else:
        utils.gems_assert_close(res, ref, dtype)


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

    _assert_conj_close(res_out, ref_out, dtype)


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

    _assert_conj_close(res_out, ref_out, dtype)
